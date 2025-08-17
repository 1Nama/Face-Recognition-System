from psycopg2.extensions import register_adapter, AsIs
from flask import Flask, jsonify, request
import psycopg2
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import numpy as np
from PIL import Image
from psycopg2 import sql
import cv2
from datetime import datetime, timedelta
import json
import requests
from io import BytesIO
import os
import warnings
from flask import render_template
import torch
from flask_cors import CORS
from psycopg2.extras import execute_batch
import base64
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

program_id = 0
model_test = AntiSpoofPredict(program_id)
image_cropper = CropImage()
model_dir="D:/Placement/Face recog temp/resources/anti_spoof_models"

mtcnn = MTCNN(min_face_size=50, thresholds=[0.6, 0.7, 0.7], keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

SIMILARITY_THRESHOLD = 0.65
LOGGING_WINDOW = timedelta(minutes=5)

# Database connection parameters
target_db = 'face_recognition'

base_conn_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'database_683',
    'host': 'localhost',
    'port': 5432
}

target_conn_params={
    'dbname': target_db,
    'user': 'postgres',
    'password': 'database_683',
    'host': 'localhost',
    'port': 5432
}

# Connect to the database
def connect_to_db():
    try:
        conn = psycopg2.connect(**target_conn_params)
        conn.autocommit = True
        cur = conn.cursor()
        return conn, cur
    except Exception as e:
        return None, None

# Add logs
def add_logs(log_level, message):
    log_date = datetime.now().strftime('%Y-%m-%d')
    log_time = datetime.now().strftime('%H:%M:%S')
    try:
        conn, cur = connect_to_db()
        if conn is None or cur is None:
            return

        cur.execute('''
            INSERT INTO logs (log_date, log_time, log_level, message)
            VALUES (%s, %s, %s, %s);
        ''', (log_date, log_time, log_level, message))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("Error adding log:", e)

# Initialize database and create necessary tables and indexes
def init_db():
    try:

        # Connect to default DB
        base_conn = psycopg2.connect(**base_conn_params)
        base_conn.autocommit = True
        base_cursor = base_conn.cursor()

        # Create database if not exists
        base_cursor.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [target_db]
        )
        if not base_cursor.fetchone():
            base_cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(target_db)
            ))

        base_cursor.close()
        base_conn.close()

        # Connect to target DB
        conn, cur = connect_to_db()

        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create student table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS student (
                college_id VARCHAR(64) NOT NULL,
                program_id VARCHAR(64) NOT NULL,
                branch_id VARCHAR(64) NOT NULL,
                student_id VARCHAR(64) NOT NULL,
                student_name VARCHAR(255) NOT NULL,
                vector_embedding VECTOR(512) NOT NULL,
                UNIQUE (college_id, program_id,branch_id,student_id)
            );
        ''')
        # Create indexes for faster queries
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_college_program_branch
            ON student (college_id, program_id, branch_id);
        ''')
        
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_student_lookup
            ON student (college_id, program_id, branch_id, student_id);
        ''')
        
        # Dynamic IVFFLAT Index
        cur.execute("SELECT COUNT(*) FROM student;")
        row_count = cur.fetchone()[0]

        if row_count < 1_000_000:
            lists = max(1, row_count // 1000)
        else:
            lists = int(row_count ** 0.5)

        cur.execute("DROP INDEX IF EXISTS idx_vector_cosine;")

        cur.execute(f'''
            CREATE INDEX idx_vector_cosine
            ON student USING ivfflat (vector_embedding vector_cosine_ops)
            WITH (lists = {lists});
        ''')

        cur.execute("ANALYZE student;")

        cur.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                log_date DATE NOT NULL,
                log_time TIME NOT NULL,
                log_level TEXT NOT NULL,
                message TEXT NOT NULL
            );
        ''')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                Date DATE NOT NULL,
                Time TIME NOT NULL,
                college_id VARCHAR(64) NOT NULL,
                program_id VARCHAR(64) NOT NULL,
                branch_id VARCHAR(64) NOT NULL,
                student_id VARCHAR(64) NOT NULL,
                student_name VARCHAR(255) NOT NULL
            );
        ''')

        add_logs("Info", "Database initialized successfully")

        cur.close()
        conn.commit()
        conn.close()

    except Exception as e:
        print("Error initializing database:", e)

init_db()

# Register adapter for numpy arrays to work with PostgreSQL vector type
def adapt_numpy_array(arr):
    return AsIs(f"'[{', '.join(map(str, arr.tolist()))}]'::vector")

register_adapter(np.ndarray, adapt_numpy_array)

def pre_augment():
    return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])

# Generate face embedding from image paths with augmentation
def get_embedding(image_paths, num_augmentations=3):
        paths=image_paths[0].split(",")

        embeddings = []
        augment = pre_augment()

        add_logs("Info", f"Starting embedding generation for images with {num_augmentations} augmentations")

        for path in paths:
            print(path)
            add_logs("Debug", f"Processing image: {path}")
            try:
                # S3 url image
                add_logs("Debug", f"Downloading image from {path}")
                #response = requests.get(path)
               # img = Image.open(BytesIO(response.content)).convert('RGB')

                # local image
                img = Image.open(path).convert('RGB')

                face = mtcnn(img)
                if face is not None:
                    add_logs("Info", "Face detected in image")
                    print("Face detected in image")
                    face = face.unsqueeze(0)
                    embedding = model(face).detach().cpu().numpy()
                    embeddings.append(embedding)
                else:
                    print("No face detected in image")
                    add_logs("Warning", "No face detected in image")

                #for i in range(num_augmentations):
                #    add_logs("Debug", f"Generating augmented image {i+1}/{num_augmentations} for image")
                #    aug_img = augment(img)
                #    face = mtcnn(aug_img)
                #    if face is not None:
                ##        face = face.unsqueeze(0)
                #        embedding = model(face).detach().cpu().numpy()
                ##        embeddings.append(embedding)

            except Exception as e:
                add_logs("Error", f"Error processing {path}: {str(e)}")

        if embeddings:
            add_logs("Info", f"Computed mean embedding from {len(embeddings)} embeddings")
            return np.mean(embeddings, axis=0).squeeze()
        else:
            add_logs("Info", "No embeddings generated, returning None")
            return None

# Match face embedding against database
def match_face(embedding, college_id, program_id, branch_id):
    conn, cur = connect_to_db()
    if conn is None or cur is None:
        add_logs("Error", "Database connection failed")
        return {"label": "Unknown", "student_id": None, "similarity": 0}

    add_logs("Info", f"Starting face match for college_id={college_id}, program_id={program_id}, branch_id={branch_id}")

    cur.execute("SELECT COUNT(*) FROM student;")
    row_count = cur.fetchone()[0]

    if row_count < 1_000_000:
        lists = max(1, row_count // 1000)
    else:
        lists = int(row_count ** 0.5)

    probes = max(1, int(lists ** 0.5))
    cur.execute(f"SET ivfflat.probes = {probes};")
    add_logs("Debug", f"Setting IVFFLAT probes to {probes}")

    cur.execute("""
            SELECT student_name, student_id, vector_embedding <=> %s AS distance
            FROM student
            WHERE college_id = %s AND program_id = %s AND branch_id = %s
            ORDER BY distance ASC
            LIMIT 1;
        """, (
            embedding,
            college_id,
            program_id,
            branch_id
        ))
    
    row = cur.fetchone()

    if row:
        student_name, student_id, distance = row
        similarity=1-distance

        if similarity >= SIMILARITY_THRESHOLD:
            return {
                    "label": student_name,
                    "student_id": student_id,
                    "similarity": similarity
                }
        else:
            return {
                    "label": "Unknown",
                    "student_id": None,
                    "similarity": similarity
                }
    else:
        add_logs("Info", "No match found in database")
        return {
                "label": "Unknown",
                "student_id": None,
                "similarity": 0
            }

# Verify face against specific student   
def compare_to_exact_student(embedding, college_id, program_id, branch_id, student_id):
    conn, cur = connect_to_db()

    if conn is None or cur is None:
        add_logs("Error", "Database connection failed")
        return {"label": "Access Denied", "student_name": None, "similarity": 0}

    add_logs("Info", f"Verifying face for college_id={college_id}, program_id={program_id}, branch_id={branch_id} ,student_id={student_id}")

    cur.execute("""
            SELECT student_name, vector_embedding <=> %s AS distance
            FROM student
            WHERE college_id = %s AND program_id = %s AND branch_id = %s AND student_id = %s;
        """, (
            embedding,         
            college_id,
            program_id,
            branch_id,
            student_id
        ))

    row = cur.fetchone()
    if row:
        student_name, distance = row
        similarity = 1 - distance 

        if similarity >= SIMILARITY_THRESHOLD:
            return {
                "label": "Access Granted",
                "student_name": student_name,
                "similarity": similarity
                }
        else:
            return {
                "label": "Access Denied",
                "student_name": None,
                "similarity": similarity
                } 
    else:
        add_logs("Warning", f"No record found for student_id={student_id}")
        return {
            "label": "Access Denied",
            "student_name": None,
            "similarity": 0
            }

# Preprocess frame for better face detection
def preprocess_frame(frame):
        add_logs("Debug", "Starting frame preprocessing")
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        processed_frame=cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        add_logs("Debug", "Frame preprocessing completed")
        return processed_frame

# Check if face is real using anti-spoofing model  
def is_real_face(frame):
    add_logs("Info", "Starting real face detection")
    try:
        image_bbox = model_test.get_bbox(frame)
        if image_bbox is None:
            add_logs("Warning", "No bounding box detected in frame")
            return False, None

        prediction = np.zeros((1, 3))
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": scale is not None,
            }
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        
        prediction = prediction / 2
        label = np.argmax(prediction)

        if prediction[0][2]>0.90:
            add_logs("Warning", "Spoof detected")
            return False,image_bbox
        else:
            add_logs("Info", "Real face confirmed")
            return True, image_bbox

    except Exception as e:
        add_logs("Error", f"Spoof detection error: {str(e)}")
        return False, None

# Update the recognition logs
def update_recognition_logs(student_data):
    try:
        conn,cur=connect_to_db()

        if conn is None or cur is None:
            add_logs("Error", "Database connection failed")
            return

        # Extract parameters
        add_logs("Debug", f"Updating recognition logs for student_data: {student_data}")
        student_id = student_data['student_id']
        college_id = student_data['college_id']
        program_id = student_data['program_id']
        branch_id = student_data['branch_id']
        student_name = student_data['student_name']
        current_date = student_data['date'] 
        current_time = student_data['time']

        current_date = datetime.strptime(student_data['date'], "%Y-%m-%d").date()
        current_time = datetime.strptime(student_data['time'], "%H:%M:%S").time()
        current_datetime = datetime.combine(current_date, current_time)

        # Get last login for this student
        query = """
            SELECT Date, Time FROM recognition_logs
            WHERE college_id = %s AND program_id = %s AND branch_id = %s AND student_id = %s
            ORDER BY Date DESC, Time DESC
            LIMIT 1;
        """
        cur.execute(query, (college_id, program_id, branch_id,student_id))
        result = cur.fetchone()

        insert_flag = False

        if result is None:
            insert_flag = True
        else:
            last_date, last_time = result
            last_log_time = datetime.combine(last_date, last_time)
            time_diff = current_datetime - last_log_time

            if time_diff >= LOGGING_WINDOW:
                insert_flag = True

        if insert_flag:
            insert_query = """
                INSERT INTO recognition_logs (college_id, program_id, branch_id, student_id, student_name, Date, Time)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            cur.execute(insert_query, (
                college_id, program_id, branch_id, student_id, student_name, current_date, current_time
            ))
            conn.commit()
            add_logs("Info","New Log inserted")
        else:
            add_logs("Info","Log skipped: within time limit")
        
        cur.close()
        conn.close()

    except Exception as e:
        add_logs("Error",f"{str(e)}")

# API for removing debugging logs
@app.route('/delete_logs',methods=['POST'])
def delete_old_records():
    try:
        conn,cur=connect_to_db()

        # Calculate cutoff date
        if conn is None or cur is None:
            add_logs("Error", "Database connection failed")
            return jsonify({"error": "Database connection failed"}), 500
        
        add_logs("Info", "Received request to delete old records from logs")
        days_ago = datetime.now() - timedelta(days=1)

        # Prepare and execute delete query
        query = f"""
            DELETE FROM logs
            WHERE log_date < %s;
        """
        cur.execute(query, (days_ago,))
        deleted_rows = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        
        add_logs("Info",f"{deleted_rows} old records deleted successfully.")
        return jsonify({"message": "old records deleted successfully."}), 200
        
    except Exception as e:
        add_logs("Error", f"Error deleting old records: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Api to clear recognition logs
@app.route('/clear-recognition-logs', methods=['POST'])
def clear_logs():
    add_logs("Info", "Received request to clear recognition logs")
    try:
        conn, cur = connect_to_db()

        delete_query = "DELETE FROM recognition_logs;" 
        cur.execute(delete_query)
        conn.commit()

        add_logs("Info", "All logs deleted from recognition_logs table")
        cur.close()
        conn.close()

        return jsonify({"message": "All logs deleted successfully"}), 200

    except Exception as e:
        add_logs("Error", f"Failed to delete logs: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API endpoint to retrieve recognition logs
@app.route('/get-recognition-logs', methods=['GET'])
def get_logs():
    add_logs("Info","Received request to retrieve recognition logs")
    try:
        conn, cur = connect_to_db()
        if conn is None or cur is None:
            add_logs("Error", "Database connection failed")
            return jsonify({"error": "Database connection failed"}), 500

        cur.execute("SELECT * FROM recognition_logs;")
        rows = cur.fetchall()

        logs = [
            {   
                "date":row[0].isoformat(),
                "time":row[1].strftime('%H:%M:%S'),
                "college_id": row[2],
                "program_id": row[3],
                "branch_id": row[4],
                "student_id": row[5],
                "student_name": row[6]
                
            }
            for row in rows
        ]
        add_logs("Info", f"Retrieved {len(logs)} logs from database")
        return jsonify(logs)
    except Exception as e:
        add_logs("Error", f"Error retrieving logs: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API endpoint to retrieve all students
@app.route('/students', methods=['GET'])
def get_students():
    add_logs("Info", "Received request to retrieve all students")
    try:
        conn, cur = connect_to_db()
        if conn is None or cur is None:
            add_logs("Error", "Database connection failed")
            return jsonify({"error": "Database connection failed"}), 500
        
        cur.execute("SELECT college_id, program_id, branch_id, student_id,student_name FROM student;")
        rows = cur.fetchall()
        students = [
            {
                "college_id": row[0],
                "program_id": row[1],
                "branch_id": row[2],
                "student_id": row[3],
                "student_name": row[4]
            }
            for row in rows
        ]
        add_logs("Info", f"Retrieved {len(students)} students from database")
        return jsonify(students)
    except Exception as e:
        add_logs("Error", f"Error retrieving students: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API endpoint to add new students   for User
@app.route('/user-add-student', methods=['POST'])
def user_add_student():
    add_logs("Info", "Received request to add students (user)")
    data = request.get_json()  
    try:     
        inserted=0
        failed=[]

        for record in data:
            add_logs("Debug", f"Processing student record: {record}")
            try:
                embedding=get_embedding([record["image_url"]])
                if embedding is None:
                    failed.append({"college_id":record["college_id"],"program_id":record["program_id"],"branch_id":record["branch_id"],"student_id": record["student_id"], "reason": "embedding not formed"})
                    add_logs("Warning", f"Failed to generate embedding for {record}")
                    continue

                conn, cur = connect_to_db()

                cur.execute("SELECT COUNT(*) FROM student;")
                row_count = cur.fetchone()[0]

                if row_count < 1_000_000:
                    lists = max(1, row_count // 1000)
                else:
                    lists = int(row_count ** 0.5)

                probes = max(1, int(lists ** 0.5))
                cur.execute(f"SET ivfflat.probes = {probes};")

                cur.execute('''
                        SELECT student_id, vector_embedding <=> %s AS distance
                        FROM student 
                        WHERE college_id = %s AND program_id = %s AND branch_id = %s AND student_id != %s
                        ORDER BY distance
                        LIMIT 1;
                    ''', (
                        embedding,
                        record['college_id'],
                        record['program_id'],
                        record['branch_id'],
                        record['student_id']
                    ))
                
                result = cur.fetchone()
            
                if result and (1 - result[1]) > 0.76:  # Convert distance to similarity
                    failed.append({
                        "college_id": record["college_id"],
                        "program_id": record["program_id"],
                        "branch_id": record["branch_id"],
                        "student_id": record["student_id"],
                        "reason": f"Too similar to existing student {result}"
                        })
                    add_logs("Warning",f"Student {record} too similar to {result}")
                    add_logs("Info",f"failed adding {record}")
                    continue
                
                cur.execute('''
                    DELETE FROM student
                    WHERE college_id = %s AND program_id = %s AND branch_id = %s AND student_id = %s;
                ''', (
                    record['college_id'],
                    record['program_id'],
                    record['branch_id'],
                    record['student_id']
                ))
                        
                cur.execute('''
                    INSERT INTO student (college_id, program_id, branch_id, student_id, student_name,vector_embedding)
                    VALUES (%s, %s, %s, %s, %s,%s);
                ''', (
                    record['college_id'],
                    record['program_id'],
                    record['branch_id'],
                    record['student_id'],
                    record['student_name'],
                    embedding
                ))

                inserted +=1
                add_logs("Info", f"Successfully added {record}")

            except Exception as e:
                failed.append({"college_id":record["college_id"],"program_id":record["program_id"],"branch_id":record["branch_id"],"student_id": record.get("student_id"), "reason": str(e)})
                add_logs("Error", f"Error adding {record}: {str(e)}")


        return jsonify({
            "message": "Students added successfully",
            "inserted": inserted,
            "failed": failed
        }),200
    
    except Exception as e:
        add_logs("Error", f"Error in user-add-student endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API endpoint to add new students    
@app.route('/admin-add-student', methods=['POST'])
def admin_add_student():
    add_logs("Info", "Received request to add students (admin)")
    data = request.get_json()  

    try:     
        inserted=0
        failed=[]

        for record in data:
            add_logs("Debug", f"Processing student record: {record}")
            try:
                embedding=get_embedding([record["image_url"]])
                if embedding is None:
                    failed.append({"college_id":record["college_id"],"program_id":record["program_id"],"branch_id":record["branch_id"],"student_id": record["student_id"], "reason": "embedding not formed"})
                    add_logs("Warning", f"Failed to generate embedding for {record}")
                    continue

                conn, cur = connect_to_db()
                cur.execute('''
                    DELETE FROM student
                    WHERE college_id = %s AND program_id = %s AND branch_id = %s AND student_id = %s;
                ''', (
                    record['college_id'],
                    record['program_id'],
                    record['branch_id'],
                    record['student_id']
                ))
                        
                cur.execute('''
                    INSERT INTO student (college_id, program_id, branch_id, student_id, student_name,vector_embedding)
                    VALUES (%s, %s, %s, %s, %s,%s);
                ''', (
                    record['college_id'],
                    record['program_id'],
                    record['branch_id'],
                    record['student_id'],
                    record['student_name'],
                    embedding
                ))

                inserted +=1
                add_logs("Info", f"Successfully added {record}")

            except Exception as e:
                add_logs("Error", f"Error adding {record}: {str(e)}")
                failed.append({"college_id":record["college_id"],"program_id":record["program_id"],"branch_id":record["branch_id"],"student_id": record.get("student_id"), "reason": str(e)})
        


        return jsonify({
            "message": "Students added successfully",
            "inserted": inserted,
            "failed": failed
        }),200
    
    except Exception as e:
        add_logs("Error", f"Error in admin-add-student endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

# API endpoint to delete students  
@app.route('/delete-student', methods=['DELETE'])
def delete_student():
    add_logs("Info", "Received request to delete students")
    data= request.get_json()

    try:
        deleted=0
        failed=[]

        for record in data:
            add_logs("Debug", f"Processing delete for {record}")
            try:
                conn, cur = connect_to_db()
                cur.execute('''
                    DELETE FROM student
                    WHERE college_id = %s AND program_id = %s AND branch_id = %s AND student_id = %s;
                ''', (
                    record['college_id'],
                    record['program_id'],
                    record['branch_id'],
                    record['student_id']
                ))

                deleted +=1
                add_logs("Info", f"Successfully deleted record {record}")

            except Exception as e:
                failed.append({"college_id":record["college_id"],"program_id":record["program_id"],"branch_id":record["branch_id"],"student_id": record.get("student_id"), "reason": str(e)})
                add_logs("Error", f"Error deleting record {record}: {str(e)}")

        return jsonify({
            "message": "Students deleted successfully",
            "deleted": deleted,
            "failed": failed
        }),200
    except Exception as e:
        add_logs("Error", f"Error in delete-student endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# API endpoint for web-based identity verification
@app.route('/web_logging1', methods=['POST'])
def live_web_logging1():
        add_logs("Info", "Received request for web logging")
        data = request.get_json()
        base64_image = data.get("image")
        college_id = data.get("college_id")
        program_id = data.get("program_id")
        branch_id = data.get("branch_id")
        student_id = data.get("student_id")
        
        if not base64_image:
            add_logs("Warning", "No image provided in request")
            return jsonify({"error":"No image provided"})

        header, encoded = base64_image.split(",", 1)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame = cv2.resize(frame, (375, 500))

        # === 1. Anti-Spoofing Check ===
        add_logs("Info", "Performing anti-spoofing check")
        is_real, _ = is_real_face(frame)
        if not is_real:
            add_logs("Warning", "Fake face detected")
            response_data = {
                "label": "Fake Face",
                "access": False,
                "student_name": None,
                "similarity": None
            }
        else:
            add_logs("Info","Real face confirmed, proceeding with preprocessing")
            frame = preprocess_frame(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame).convert('RGB')

            face = mtcnn(img)
            boxes, _ = mtcnn.detect(img)
            
            if face is not None and boxes is not None:
                add_logs("Info","Face detected, generating embedding")
                face = face.unsqueeze(0)
                embedding = model(face).detach().cpu().numpy().squeeze()

                add_logs("Debug", f"Comparing embedding for {data}")
                df=compare_to_exact_student(embedding,college_id=college_id, program_id=program_id, branch_id=branch_id, student_id=student_id)
                
                label= df['label']
                student_name= df['student_name']
                similarity= df['similarity']

                if label == "Access Granted":
                    add_logs("Info", f"Access granted for {data}, similarity={similarity:.4f}")
                    response_data = {
                    "label": label,
                    "access":True,
                    "student_name": student_name,
                    "similarity": float(similarity)  
                }
                else:
                    add_logs("Warning", f"Access denied because low similarity={similarity:.4f}")
                    response_data = {
                        "label": label,
                        "access":False,
                        "student_name": None,
                        "similarity": None
                    }

                
            else:
                add_logs("Warning", "No face detected in image")
                response_data = {
                    "label": "No Face Detected",
                    "access": False,
                    "student_name": None,
                    "similarity": None
                } 

        return jsonify(response_data)

# API endpoint for web-based random face recognition
@app.route('/web_random', methods=['POST'])
def live_random_recognize1():
        add_logs("Info", "Received request for web random recognition")
        data= request.get_json()
        base64_image = data.get("image")
        college_id = data.get("college_id")
        program_id = data.get("program_id")
        branch_id = data.get("branch_id")
        
        if not base64_image:
            add_logs("Warning", "No image provided in request")
            return jsonify({"error":"No image provided"})

        header, encoded = base64_image.split(",", 1)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


        frame = cv2.resize(frame, (375, 500))

        add_logs("Info", "Performing anti-spoofing check")
        is_real, image_bbox = is_real_face(frame)

        if not is_real:
            add_logs("Warning", "Fake face detected")
            response_data = {
                "label": "Fake Face",
                "student_id": None,
                "similarity": None
                }     
        else:
            add_logs("Info", "Real face confirmed, proceeding with preprocessing")
            frame = preprocess_frame(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame).convert('RGB')

            face = mtcnn(img)
            boxes, _ = mtcnn.detect(img, landmarks=False)

            if face is not None and boxes is not None:
                add_logs("Info", "Face detected, generating embedding")
                face = face.unsqueeze(0)
                embedding = model(face).detach().cpu().numpy().squeeze()

                add_logs("Debug", f"Matching embedding for college_id={college_id}, program_id={program_id}, branch_id={branch_id}")
                df=match_face(embedding,college_id=college_id,program_id=program_id,branch_id=branch_id)
                
                label= df['label']
                student_id= df['student_id']
                similarity= df['similarity']

                if label!="Unknown":
                    current_time = datetime.now()
                    update_recognition_logs({
                            "college_id": college_id,
                            "program_id": program_id,
                            "branch_id": branch_id,
                            "student_id": student_id,
                            "student_name": label,
                            "date": current_time.strftime("%Y-%m-%d"),
                            "time": current_time.strftime("%H:%M:%S")
                            })

                    add_logs("Info", f"Matched student_id={student_id}, similarity={similarity:.4f}")
                    response_data = {
                        "label": label,
                        "student_id": student_id,
                        "similarity": float(similarity)  
                    }
                else:
                    add_logs("Warning", f"No match found, label=Unknown, similarity={similarity:.4f}")
                    response_data = {
                        "label": label,
                        "student_id": student_id,
                        "similarity": float(similarity)
                    }
            else:
                add_logs("Warning", "No face detected in image")
                response_data = {
                    "label": "No Face Detected",
                    "student_id": None,
                    "similarity": None
                }

        return jsonify(response_data)


if __name__ == '__main__': 
    app.run(debug=True)
