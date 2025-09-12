from flask import Flask, render_template, request, send_file, url_for, flash, session, redirect
import psycopg2
import re
import os
import base64
from collections import defaultdict
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from dotenv import load_dotenv


app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# PostgreSQL connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT')
    )


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


if __name__ == '__main__':
    app.run(debug=True)