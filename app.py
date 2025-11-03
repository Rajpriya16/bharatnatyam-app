from flask import Flask, render_template, request, send_file, url_for, flash, session, redirect
import psycopg2
import re
import os
import base64
from collections import defaultdict
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from dotenv import load_dotenv
import cv2
import numpy as np
import mediapipe as mp
from werkzeug.utils import secure_filename
from rembg import remove
from PIL import Image
import io
import pickle


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = os.getenv('FLASK_SECRET_KEY')

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


# PostgreSQL connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT')
    )

@app.template_filter('b64decode')
def b64decode_filter(encoded):
    try:
        return base64.urlsafe_b64decode(encoded.encode()).decode()
    except Exception:
        return "[Invalid path]"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# Extract frame number from filename
def extract_frame_number(file_path):
    match = re.search(r'_(\d+)\.png$', file_path)
    return int(match.group(1)) if match else None

# Encode and decode file paths safely
def encode_path(path):
    return base64.urlsafe_b64encode(path.encode()).decode()

def decode_path(encoded):
    return base64.urlsafe_b64decode(encoded.encode()).decode()

# Extract dancer number from code string
def extract_dancer_number(code):
    if isinstance(code, int):
        return code
    match = re.search(r'D(\d+)', str(code))
    return int(match.group(1)) if match else None

# Route to serve image from disk
@app.route('/serve_image/<encoded_path>')
def serve_image(encoded_path):
    try:
        file_path = decode_path(encoded_path)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            print("File not found:", file_path)
    except Exception as e:
        print("Error decoding path:", e)
    return "Image not found", 404

@app.route('/serve_video/<encoded_path>')
def serve_video(encoded_path):
    try:
        file_path = decode_path(encoded_path)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='video/avi')
    except Exception as e:
        print("Error serving video:", e)
    return "Video not found", 404

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except psycopg2.IntegrityError as e:
            conn.rollback()
            if 'unique constraint' in str(e).lower():
                flash('Username already exists. Please choose another.', 'danger')
            else:
                flash('An error occurred. Please try again.', 'danger')
        finally:
            cur.close()
            conn.close()
    return render_template('signup.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user[0], password):
            session['username'] = username
            return redirect(url_for('search_video'))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))



@app.route('/view-video', methods=['GET', 'POST'])
@login_required
def search_video():
    video_info = []
    error = None

    dance_types = {
        1: "tatta", 2: "natta", 3: "pakka", 4: "kuditta_mettu", 5: "kuditta_nattal",
        6: "kuditta_tattal", 7: "paikal", 8: "tei_tei_dhatta", 9: "katti_kartari",
        10: "uttsanga", 11: "mandi", 12: "sarrikkal", 13: "trimana"
    }

    # Fetch all sequences grouped by dance_type_id
    conn = get_db_connection()
    cur = conn.cursor()
    sequence_map = {}

    try:
        cur.execute("SELECT dance_type_id, sequence_number FROM Sequence ORDER BY dance_type_id, sequence_number")
        rows = cur.fetchall()
        for dance_id, seq_num in rows:
            sequence_map.setdefault(dance_id, []).append(seq_num)
    except Exception as e:
        error = f"Error loading sequences: {e}"
    finally:
        cur.close()
        conn.close()

    if request.method == 'POST':
        dancetype = request.form['dancetype']
        sequence_id = request.form['sequence_id']
        dancer_id = request.form['dancer_id']

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT video_path FROM videos
                WHERE dancetype = %s AND sequence_id = %s AND dancer_id = %s
            """, (dancetype, sequence_id, dancer_id))

            results = cur.fetchall()
            for row in results:
                raw_path = row[0]
                filename = os.path.basename(raw_path)
                type_match = re.match(r'(color|depth|skeleton)', filename)
                video_type = type_match.group(1).capitalize() if type_match else "Unknown"
                video_info.append({
                    "encoded_path": encode_path(raw_path),
                    "video_type": video_type
                })


        except Exception as e:
            error = f"Database error: {e}"

        finally:
            cur.close()
            conn.close()

    return render_template(
        'video_search.html',
        video_info=video_info,
        error=error,
        dance_types=dance_types,
        sequence_map=sequence_map
    )

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/view-color', methods=['GET', 'POST'])
@login_required
def view_color_frames():
    frames = []
    user_input_code = ''
    error = None

    if request.method == 'POST':
        user_input_code = request.form['query_code'].strip()

        conn = get_db_connection()
        cur = conn.cursor()
        #print("************************************************************")
        try:
            cur.execute("""
                INSERT INTO posture_frequency (posture, frequency)
                VALUES (%s, 1)
                ON CONFLICT (posture)
                DO UPDATE SET frequency = posture_frequency.frequency + 1;
            """, (user_input_code,))
            print("************************************************************")
            conn.commit()
        except Exception as freq_error:
            print("Frequency update error:", freq_error)


        try:
            posture_ranges=None
            if re.fullmatch(r'P\d+', user_input_code):
                cur.execute("""
                    SELECT kp_start, kp_end, query_code 
                    FROM posture_query 
                    WHERE query_code LIKE %s
                """, ('%' + user_input_code,))
                posture_ranges = cur.fetchall()

            if not posture_ranges:
                error = "No matching posture_query entries found."
                raise Exception()

            # Fetch all frames
            cur.execute("""
                SELECT file_path, dancer_id, frame_type_id 
                FROM Frame 
                ORDER BY file_path
            """)
            raw_frames = cur.fetchall()

            seen_per_dancer = defaultdict(set)

            for file_path, dancer_id, frame_type_id in raw_frames:
                extracted_number = extract_frame_number(file_path)
                if extracted_number is None or frame_type_id != 1:
                    continue

                frame_dancer_num = extract_dancer_number(dancer_id)

                for kp_start, kp_end, query_code in posture_ranges:
                    query_dancer_num = extract_dancer_number(query_code)

                    if (
                        kp_start <= extracted_number <= kp_end and
                        query_dancer_num == frame_dancer_num and
                        extracted_number not in seen_per_dancer[frame_dancer_num]
                    ):
                        frames.append({
                            "frame_number": extracted_number,
                            "file_path": encode_path(file_path),
                            "frame_type": 'color',
                            "dancer_code": f"Dancer {frame_dancer_num}"
                        })
                        seen_per_dancer[frame_dancer_num].add(extracted_number)

        except Exception as e:
            print("Error:", e)

        cur.close()
        conn.close()

    # Group frames by dancer
    grouped_frames = defaultdict(list)
    for frame in frames:
        grouped_frames[frame["dancer_code"]].append(frame)

    return render_template('color_display.html',frames=frames,query_code=user_input_code,error=error,grouped_frames=grouped_frames)

@app.route('/view-skeleton', methods=['GET', 'POST'])
@login_required
def view_skeleton_frames():
    frames = []
    user_input_code = ''
    error = None

    if request.method == 'POST':
        user_input_code = request.form['query_code'].strip()

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO posture_frequency (posture, frequency)
                VALUES (%s, 1)
                ON CONFLICT (posture)
                DO UPDATE SET frequency = posture_frequency.frequency + 1;
            """, (user_input_code,))
            print("************************************************************")
            conn.commit()
        except Exception as freq_error:
            print("Frequency update error:", freq_error)

        try:
            posture_ranges=None
            if re.fullmatch(r'P\d+', user_input_code):
                cur.execute("""
                    SELECT kp_start, kp_end, query_code 
                    FROM posture_query 
                    WHERE query_code LIKE %s
                """, ('%' + user_input_code,))
                posture_ranges = cur.fetchall()

            if not posture_ranges:
                error = "No matching posture_query entries found."
                raise Exception()

            # Fetch all frames
            cur.execute("""
                SELECT file_path, dancer_id, frame_type_id 
                FROM Frame 
                ORDER BY file_path
            """)
            raw_frames = cur.fetchall()

            seen_per_dancer = defaultdict(set)

            for file_path, dancer_id, frame_type_id in raw_frames:
                extracted_number = extract_frame_number(file_path)
                if extracted_number is None or frame_type_id != 2:
                    continue

                frame_dancer_num = extract_dancer_number(dancer_id)

                for kp_start, kp_end, query_code in posture_ranges:
                    query_dancer_num = extract_dancer_number(query_code)

                    if (
                        kp_start <= extracted_number <= kp_end and
                        query_dancer_num == frame_dancer_num and
                        extracted_number not in seen_per_dancer[frame_dancer_num]
                    ):
                        frames.append({
                            "frame_number": extracted_number,
                            "file_path": encode_path(file_path),
                            "frame_type": 'color',
                            "dancer_code": f"Dancer {frame_dancer_num}"
                        })
                        seen_per_dancer[frame_dancer_num].add(extracted_number)

        except Exception as e:
            print("Error:", e)

        cur.close()
        conn.close()

    # Group frames by dancer
    grouped_frames = defaultdict(list)
    for frame in frames:
        grouped_frames[frame["dancer_code"]].append(frame)

    return render_template('skeleton_display.html',frames=frames,query_code=user_input_code,error=error,grouped_frames=grouped_frames)

@app.route('/view-depth', methods=['GET', 'POST'])
@login_required
def view_depth_frames():
    frames = []
    user_input_code = ''
    error = None

    if request.method == 'POST':
        user_input_code = request.form['query_code'].strip()

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO posture_frequency (posture, frequency)
                VALUES (%s, 1)
                ON CONFLICT (posture)
                DO UPDATE SET frequency = posture_frequency.frequency + 1;
            """, (user_input_code,))
            print("************************************************************")
            conn.commit()
        except Exception as freq_error:
            print("Frequency update error:", freq_error)

        try:
            posture_ranges=None
            if re.fullmatch(r'P\d+', user_input_code):
                cur.execute("""
                    SELECT kp_start, kp_end, query_code 
                    FROM posture_query 
                    WHERE query_code LIKE %s
                """, ('%' + user_input_code,))
                posture_ranges = cur.fetchall()

            if not posture_ranges:
                error = "No matching posture_query entries found."
                raise Exception()

            # Fetch all frames
            cur.execute("""
                SELECT file_path, dancer_id, frame_type_id 
                FROM Frame 
                ORDER BY file_path
            """)
            raw_frames = cur.fetchall()

            seen_per_dancer = defaultdict(set)

            for file_path, dancer_id, frame_type_id in raw_frames:
                extracted_number = extract_frame_number(file_path)
                if extracted_number is None or frame_type_id != 3:
                    continue

                frame_dancer_num = extract_dancer_number(dancer_id)

                for kp_start, kp_end, query_code in posture_ranges:
                    query_dancer_num = extract_dancer_number(query_code)

                    if (
                        kp_start <= extracted_number <= kp_end and
                        query_dancer_num == frame_dancer_num and
                        extracted_number not in seen_per_dancer[frame_dancer_num]
                    ):
                        frames.append({
                            "frame_number": extracted_number,
                            "file_path": encode_path(file_path),
                            "frame_type": 'color',
                            "dancer_code": f"Dancer {frame_dancer_num}"
                        })
                        seen_per_dancer[frame_dancer_num].add(extracted_number)

        except Exception as e:
            print("Error:", e)

        cur.close()
        conn.close()

    # Group frames by dancer
    grouped_frames = defaultdict(list)
    for frame in frames:
        grouped_frames[frame["dancer_code"]].append(frame)

    return render_template('depth_display.html',frames=frames,query_code=user_input_code,error=error,grouped_frames=grouped_frames)

@app.route('/posture-browser', methods=['GET', 'POST'])
@login_required
def posture_browser():
    posture_numbers = []
    selected_dance_id = None
    selected_dance_name = None
    error = None

    # Static list of dance types
    dance_types = {
        1: 'tatta', 2: 'natta', 3: 'pakka', 4: 'kuditta_mettu', 5: 'kuditta_nattal',
        6: 'kuditta_tattal', 7: 'paikal', 8: 'tei_tei_dhatta', 9: 'katti_kartari',
        10: 'uttsanga', 11: 'mandi', 12: 'sarrikkal', 13: 'trimana'
    }

    if request.method == 'POST':
        selected_dance_id = int(request.form['dance_id'])
        selected_dance_name = dance_types.get(selected_dance_id)

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            # Get dance_code from DanceType table
            cur.execute("SELECT dance_code FROM DanceType WHERE id = %s", (selected_dance_id,))
            result = cur.fetchone()

            if not result:
                error = "Dance code not found for selected dance type."
            else:
                dance_code = result[0]

                cur.execute("SELECT query_code FROM posture_query")
                codes = cur.fetchall()

                for (code,) in codes:
                    # Extract prefix before first digit
                    prefix_match = re.match(r'^([A-Z]+)', code)
                    if not prefix_match:
                        continue

                    code_prefix = prefix_match.group(1)
                    if code_prefix != dance_code:
                        continue  # skip mismatched codes

                    # Extract posture number from end
                    posture_match = re.search(r'P(\d+)$', code)
                    if posture_match:
                        posture_numbers.append(int(posture_match.group(1)))

                posture_numbers = sorted(set(posture_numbers))

        except Exception as e:
            error = f"Database error: {e}"

        finally:
            cur.close()
            conn.close()

    return render_template(
        'posture_browser.html',
        dance_types=dance_types,
        selected_dance_id=selected_dance_id,
        selected_dance_name=selected_dance_name,
        posture_numbers=posture_numbers,
        error=error
    )

# --- Display All Postures ---
@app.route('/create-sequence', methods=['GET', 'POST'])
@login_required
def create_sequence():
    if 'username' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))

    conn = get_db_connection()
    cur = conn.cursor()

    if request.method == 'POST':
        sequence_name = request.form['sequence_name']
        posture_ids = request.form.getlist('posture_ids[]')  # List of selected postures
        posture_ids = [int(pid) for pid in posture_ids]  # convert to integers
        username = session['username']

        # Insert sequence using username instead of user_id
        cur.execute("""
            INSERT INTO user_sequence (user_id, sequence_name)
            SELECT id, %s FROM users WHERE username = %s
            RETURNING id;
        """, (sequence_name, username))
        sequence_id = cur.fetchone()[0]

        # Insert into user_sequence_postures
        for order_num, pid in enumerate(posture_ids, start=1):
            cur.execute("""
                INSERT INTO user_sequence_postures (sequence_id, posture_id, order_num)
                VALUES (%s, %s, %s);
            """, (sequence_id, pid, order_num))

        conn.commit()
        flash("Sequence saved successfully!")
        cur.close()
        conn.close()
        return redirect(url_for('view_sequences'))

    # GET request: fetch all posture images
    cur.execute("SELECT id, posture_code, frame_path FROM posture_image ORDER BY id;")
    postures = cur.fetchall()
    cur.close()
    conn.close()

    return render_template('create_sequence.html', postures=postures, encode_path=encode_path)


# --- Save User-Created Sequence ---
@app.route('/save_sequence', methods=['POST'])
@login_required
def save_sequence():
    if 'username' not in session:
        flash("Please log in first.")
        return redirect(url_for('login'))
    
    sequence_name = request.form['sequence_name']
    selected_order = request.form.getlist('posture_order')  # ordered list of posture IDs
    username = session['username']

    conn = get_db_connection()
    cur = conn.cursor()

    # Get user_id from username
    cur.execute("SELECT id FROM users WHERE username = %s;", (username,))
    user_row = cur.fetchone()
    if not user_row:
        flash("User not found.")
        cur.close()
        conn.close()
        return redirect(url_for('login'))
    user_id = user_row[0]

    # Insert into user_sequence
    cur.execute("""
        INSERT INTO user_sequence (user_id, sequence_name)
        VALUES (%s, %s)
        RETURNING id;
    """, (user_id, sequence_name))
    sequence_id = cur.fetchone()[0]

    # Insert each posture in order
    for pos, posture_id in enumerate(selected_order, start=1):
        cur.execute("""
            INSERT INTO user_sequence_postures (sequence_id, posture_id, order_num)
            VALUES (%s, %s, %s);
        """, (sequence_id, posture_id, pos))

    conn.commit()
    cur.close()
    conn.close()

    return redirect(url_for('view_sequences'))


# --- View User's Saved Sequences ---
@app.route('/my-sequences')
@login_required
def view_sequences():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    conn = get_db_connection()
    cur = conn.cursor()

    # Get sequences for this user
    cur.execute("""
        SELECT us.id, us.sequence_name, us.created_at
        FROM user_sequence us
        JOIN users u ON us.user_id = u.id
        WHERE u.username = %s
        ORDER BY us.created_at DESC;
    """, (username,))
    sequences_data = cur.fetchall()

    sequences = []
    for seq_id, seq_name, created_at in sequences_data:
        # Get postures for each sequence in order
        cur.execute("""
            SELECT p.id, p.frame_path
            FROM user_sequence_postures usp
            JOIN posture_image p ON usp.posture_id = p.id
            WHERE usp.sequence_id = %s
            ORDER BY usp.order_num ASC;
        """, (seq_id,))
        postures = cur.fetchall()  # list of (posture_id, frame_path)
        sequences.append({
            'id': seq_id,
            'name': seq_name,
            'created_at': created_at,
            'postures': postures
        })

    cur.close()
    conn.close()

    return render_template('my_sequences.html', sequences=sequences, encode_path=encode_path)

pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

def get_reference_postures():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT posture_code, dance_form, frame_path FROM posture_image;")
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def extract_keypoints(image_path):
    with open(image_path, 'rb') as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes)
    image_no_bg = Image.open(io.BytesIO(output_bytes)).convert("RGB")
    image_np = np.array(image_no_bg)

    results = pose.process(image_np)
    if not results.pose_landmarks:
        return None

    keypoints = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])
    return normalize_keypoints(keypoints)

def normalize_keypoints(keypoints):
    if keypoints is None or len(keypoints) == 0:
        return None

    mid_hip = (keypoints[23] + keypoints[24]) / 2
    centered = keypoints - mid_hip

    shoulder_dist = np.linalg.norm(keypoints[11] - keypoints[12])
    if shoulder_dist == 0:
        shoulder_dist = 1.0
    normalized = centered / shoulder_dist

    return normalized

def compare_keypoints(user_kp, ref_kp):
    if user_kp is None or ref_kp is None or len(user_kp) != len(ref_kp):
        return float('inf')

    u = user_kp.flatten()
    r = ref_kp.flatten()

    cos_sim = np.dot(u, r) / (np.linalg.norm(u) * np.linalg.norm(r))
    return 1 - cos_sim

def generate_reference_cache():
    reference_data = get_reference_postures()
    cache = []

    for posture_code, dance_form, frame_path in reference_data:
        ref_kp = extract_keypoints(frame_path)
        print(posture_code,"\n")
        if ref_kp is not None:
            cache.append((posture_code, dance_form, frame_path, ref_kp))

    with open('reference_keypoints.pkl', 'wb') as f:
        pickle.dump(cache, f)

@app.route('/upload-posture', methods=['GET', 'POST'])
def upload():
    matches = []
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            user_kp = extract_keypoints(filepath)

            with open('reference_keypoints.pkl', 'rb') as f:
                reference_data = pickle.load(f)

            comparisons = []
            for posture_code, dance_form, frame_path, ref_kp in reference_data:
                score = compare_keypoints(user_kp, ref_kp)
                comparisons.append((score, posture_code, dance_form, frame_path))

            top_matches = sorted(comparisons, key=lambda x: x[0])[:5]
            max_score = max([s for s, _, _, _ in comparisons if s != float('inf')], default=1)

            for score, posture_code, dance_form, frame_path in top_matches:
                similarity = max(0, 100 - (score / max_score) * 100)
                matches.append({
                    'score': round(score, 4),
                    'similarity': round(similarity, 2),
                    'code': posture_code,
                    'dance': dance_form,
                    'path': encode_path(frame_path)
                })

    return render_template('upload.html', matches=matches)



# @app.route('/upload-video', methods=['GET', 'POST'])
# @login_required
# def upload_video():
#     frame_matches = []

#     if request.method == 'POST':
#         file = request.files['video']
#         if file:
#             filename = secure_filename(file.filename)
#             video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(video_path)

#             # ✅ Load precomputed reference keypoints
#             with open('reference_keypoints.pkl', 'rb') as f:
#                 reference_data = pickle.load(f)

#             cap = cv2.VideoCapture(video_path)
#             frame_index = 0

#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Process every 10th frame
#                 if frame_index % 10 == 0:
#                     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     results = pose.process(image_rgb)

#                     if results.pose_landmarks:
#                         keypoints = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])
#                         user_kp = normalize_keypoints(keypoints)

#                         comparisons = []
#                         for posture_code, dance_form, frame_path, ref_kp in reference_data:
#                             score = compare_keypoints(user_kp, ref_kp)
#                             comparisons.append((score, posture_code, dance_form, frame_path))

#                         top_match = min(comparisons, key=lambda x: x[0])
#                         similarity = max(0, 100 - top_match[0] * 100)

#                         # Save current frame as image
#                         frame_filename = f"frame_{frame_index}.png"
#                         saved_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
#                         cv2.imwrite(saved_frame_path, frame)

#                         frame_matches.append({
#                             'frame': frame_index,
#                             'code': top_match[1],
#                             'dance': top_match[2],
#                             'similarity': round(similarity, 2),
#                             'path': encode_path(top_match[3]),         # matched posture image
#                             'frame_path': encode_path(saved_frame_path) # uploaded video frame
#                         })

#                 frame_index += 1

#             cap.release()

#     return render_template('video_results.html', matches=frame_matches)


@app.route('/upload-video', methods=['GET', 'POST'])
@login_required
def upload_video():
    frame_matches = []

    if request.method == 'POST':
        file = request.files['video']
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            with open('reference_keypoints.pkl', 'rb') as f:
                reference_data = pickle.load(f)

            cap = cv2.VideoCapture(video_path)
            frame_index = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every 10th frame
                if frame_index % 10 == 0:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    if results.pose_landmarks:
                        keypoints = np.array([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])
                        user_kp = normalize_keypoints(keypoints)

                        comparisons = []
                        for posture_code, dance_form, frame_path, ref_kp in reference_data:
                            score = compare_keypoints(user_kp, ref_kp)
                            comparisons.append((score, posture_code, dance_form, frame_path))

                        top_match = min(comparisons, key=lambda x: x[0])
                        similarity = max(0, 100 - top_match[0] * 100)

                        # Save current frame as image
                        frame_filename = f"frame_{frame_index}.png"
                        saved_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                        cv2.imwrite(saved_frame_path, frame)

                        frame_matches.append({
                            'frame': frame_index,
                            'code': top_match[1],
                            'dance': top_match[2],
                            'similarity': round(similarity, 2),
                            'path': encode_path(top_match[3]),         # matched posture image
                            'frame_path': encode_path(saved_frame_path) # uploaded video frame
                        })

                frame_index += 1

            cap.release()

            # 🧠 Keep only the best match per unique posture code
            unique_matches = {}
            for match in frame_matches:
                code = match['code']
                if code not in unique_matches or match['similarity'] > unique_matches[code]['similarity']:
                    unique_matches[code] = match

            # Final list of matches with unique postures
            frame_matches = list(unique_matches.values())

            # Optional: sort by similarity descending
            frame_matches.sort(key=lambda x: x['similarity'], reverse=True)

    return render_template('video_results.html', matches=frame_matches)

@app.route('/skeletal-visualization', methods=['GET', 'POST'])
@login_required
def skeletal_visualization():
    video_url = None

    if request.method == 'POST':
        file = request.files['video']
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'skeletal_overlay.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            total_frames = 0
            pose_frames = 0


            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames+=1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                    pose_frames += 1


                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )

                out.write(frame)

            cap.release()
            out.release()
            print(total_frames," ",pose_frames)
            video_url = encode_path(output_path)

    return render_template('skeletal_visualization.html', video_url=video_url)



def draw_motion_paths(keypoint_sequence, canvas_size=(640, 480)):
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    num_landmarks = keypoint_sequence[0].shape[0]

    # Assign colors to landmark groups (e.g., arms, legs, torso)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]

    for i in range(num_landmarks):
        color = colors[i % len(colors)]
        for f in range(1, len(keypoint_sequence)):
            pt1 = keypoint_sequence[f-1][i]
            pt2 = keypoint_sequence[f][i]

            # Scale normalized coordinates to canvas size
            x1, y1 = int(pt1[0] * canvas_size[0]), int(pt1[1] * canvas_size[1])
            x2, y2 = int(pt2[0] * canvas_size[0]), int(pt2[1] * canvas_size[1])

            # Draw motion path
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)

    return canvas


@app.route('/dance-slideshow', methods=['GET', 'POST'])
@login_required
def slideshow():
    conn = get_db_connection()
    cur = conn.cursor()
    BASE_IMAGE_PATH = r"G:\Dataset\Session 1 Data_only Images"

    cur.execute("SELECT id, name FROM DanceType ORDER BY name;")
    dance_forms = cur.fetchall()

    selected_dance = request.form.get('dance_form')
    selected_sequence = request.form.get('sequence_number')
    selected_dancer = request.form.get('dancer')

    sequences = []
    dancers = [(1, 'Dancer1'), (2, 'Dancer2'), (3, 'Dancer3')]
    images = []

    if selected_dance:
        cur.execute("SELECT sequence_number FROM Sequence WHERE dance_type_id = %s ORDER BY sequence_number;", (selected_dance,))
        sequences = [row[0] for row in cur.fetchall()]

    if selected_dance and selected_sequence and selected_dancer:
        cur.execute("SELECT name FROM DanceType WHERE id = %s;", (selected_dance,))
        dance_name = cur.fetchone()[0]
        folder_name = f"{selected_dance}_{dance_name}"
        dancer_folder = f"Dancer{selected_dancer}"
        sequence_folder = str(selected_sequence)

        full_path = os.path.join(BASE_IMAGE_PATH, folder_name, sequence_folder, dancer_folder)
        if os.path.exists(full_path):
            all_files = os.listdir(full_path)
            pattern = re.compile(r"color_.*_([0-9]+)\.png$")
            frame_files = []

            for f in all_files:
                match = pattern.search(f)
                if match:
                    frame_num = int(match.group(1))
                    if frame_num % 5 == 0:
                        abs_path = os.path.join(full_path, f)
                        frame_files.append((frame_num, encode_path(abs_path)))

            # Sort numerically by frame number
            images = [encoded for frame_num, encoded in sorted(frame_files, key=lambda x: x[0])]

    cur.close()
    conn.close()

    return render_template('dance_slideshow.html',
                           dance_forms=dance_forms,
                           sequences=sequences,
                           dancers=dancers,
                           selected_dance=selected_dance,
                           selected_sequence=selected_sequence,
                           selected_dancer=selected_dancer,
                           images=images)


@app.route('/submit-public-image', methods=['GET', 'POST'])
@login_required
def submit_public_image():
    message = None
    if request.method == 'POST':
        file = request.files['image']
        caption = request.form.get('caption', '').strip()

        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            conn = get_db_connection()
            cur = conn.cursor()
            # Get user ID from session
            cur.execute("SELECT id FROM users WHERE username = %s", (session['username'],))
            user_id = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO public_image_requests (user_id, image_path, caption)
                VALUES (%s, %s, %s);
            """, (user_id, save_path, caption))
            conn.commit()
            cur.close()
            conn.close()

            message = "Your image has been submitted for admin approval."

    return render_template('submit_public_image.html', message=message)

@app.route('/public-gallery')
@login_required
def public_gallery():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT pir.image_path, pir.caption, pir.approved_at, u.username
        FROM public_image_requests pir
        JOIN users u ON pir.user_id = u.id
        WHERE pir.is_approved = TRUE
        ORDER BY pir.approved_at DESC;
    """)
    raw_images = cur.fetchall()
    cur.close()
    conn.close()

    images = []
    for image_path, caption, approved_at, username in raw_images:
        images.append({
            'path': encode_path(image_path),  # ✅ encode here
            'caption': caption,
            'approved_at': approved_at,
            'username': username
        })

    return render_template('public_gallery.html', images=images)

#admin

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'rajpriya' and password == 'Asdf1234@':
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template('admin_login.html', error=error)

@app.route('/popular-postures')
@admin_required
def popular_postures():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Fetch top 10 posture codes by frequency
        cur.execute("""
            SELECT posture, frequency
            FROM posture_frequency
            ORDER BY frequency DESC
            LIMIT 10;
        """)
        freq_data = cur.fetchall()

        postures = []
        for posture_code, frequency in freq_data:
            # Find matching posture_image where posture_code ends with the posture
            cur.execute("""
                SELECT frame_path, dance_form
                FROM posture_image
                WHERE posture_code ~ %s
                LIMIT 1;
            """, (f'P{posture_code.split("P")[-1]}$',))  # Regex: ends with P<number>

            result = cur.fetchone()
            if result:
                frame_path, dance_form = result
                postures.append({
                    'code': posture_code,
                    'frequency': frequency,
                    'dance': dance_form,
                    'path': encode_path(frame_path)
                })

    except Exception as e:
        print("Error fetching popular postures:", e)
        postures = []

    cur.close()
    conn.close()

    return render_template('popular_postures.html', postures=postures)

# Admin dashboard route
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT posture_code, dance_form, frame_path FROM posture_image ORDER BY id DESC LIMIT 10;")
    raw_postures = cur.fetchall()
    cur.close()
    conn.close()

    postures = []
    for code, dance, path in raw_postures:
        postures.append({
            'code': code,
            'dance': dance,
            'path': encode_path(path)  
        })

    return render_template('admin_dashboard.html', postures=postures)

# Logout route
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/public-image-requests')
@admin_required
def view_public_image_requests():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT pir.id, pir.image_path, pir.caption, pir.submitted_at, u.username
        FROM public_image_requests pir
        JOIN users u ON pir.user_id = u.id
        WHERE pir.is_approved = FALSE
        ORDER BY pir.submitted_at DESC;
    """)
    raw_requests = cur.fetchall()
    requests = []
    for req in raw_requests:
        request_id, image_path, caption, submitted_at, username = req
        requests.append({
            'id': request_id,
            'path': encode_path(image_path), 
            'caption': caption,
            'submitted_at': submitted_at,
            'username': username
        })
    cur.close()
    conn.close()

    return render_template('admin_public_requests.html', requests=requests)

@app.route('/admin/approve-image/<int:request_id>', methods=['POST'])
@admin_required
def approve_public_image(request_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE public_image_requests
        SET is_approved = TRUE, approved_at = CURRENT_TIMESTAMP
        WHERE id = %s;
    """, (request_id,))
    conn.commit()
    cur.close()
    conn.close()
    return redirect(url_for('view_public_image_requests'))

if __name__ == '__main__':
    #generate_reference_cache()
    app.run(debug=True)