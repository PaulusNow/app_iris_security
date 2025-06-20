# Iris Security System

## Description
This project is an Iris Recognition Security System implemented as a Flask web application. It uses computer vision and machine learning techniques to scan and recognize iris patterns for user authentication. The system leverages a TensorFlow/Keras model for eye/no-eye detection, OpenCV for image processing, AES encryption for secure storage of iris templates, and MySQL as the backend database.

## Features
- Real-time iris scanning using webcam video feed.
- Eye/no-eye detection using a pre-trained AlexNet model.
- Iris feature extraction using image processing and Gabor filters.
- Secure storage of iris templates and masks with AES encryption.
- User enrollment with unique username registration.
- Iris matching with configurable similarity threshold.
- Audit logging of system events and user activities.
- Admin dashboard for managing users and viewing audit logs.
- RESTful API endpoints for scanning, enrollment, and admin operations.

## Installation

### Prerequisites
- Python 3.8 or higher
- MySQL server

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd iris_security
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare MySQL database:
   - Create a database named `iris_security` (or your preferred name).
   - Ensure the MySQL user has appropriate permissions.

5. Configure environment variables (optional, defaults are provided):
   - `MYSQL_HOST` (default: localhost)
   - `MYSQL_USER` (default: iris_app)
   - `MYSQL_PASSWORD` (default: password_kuat123!)
   - `MYSQL_DB` (default: iris_security)
   - `AES_KEY` (default: my_super_secret_key_32bytes)
   - `IRIS_MATCH_THRESHOLD` (default: 0.35)
   - `AUDIT_LOG_FILE` (default: audit.log)
   - `DEBUG_MODE` (default: False)

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Access the web interface:
   - Open your browser and go to `http://localhost:5000/` for the main page.
   - Admin dashboard available at `http://localhost:5000/admin`.

3. Use the webcam to scan iris or enroll new users via the web interface.

## Project Structure

```
iris_security/
│
├── app.py                  # Main Flask application and routes
├── iris_processing.py      # Iris image processing utilities
├── encryption.py           # (If present) Encryption utilities
├── model/                  # Pre-trained model files
│   └── best_iris_alexnet.h5
├── static/                 # Static assets (JS, CSS)
├── templates/              # HTML templates for web pages
│   ├── index.html
│   ├── admin.html
│   └── result.html
├── requirements.txt        # Python dependencies
└── audit.log               # Audit log file (if used)
```

## Notes
- The system uses AES-GCM encryption to securely store iris templates in the database.
- The iris matching threshold can be adjusted via environment variables.
- The application initializes the database tables automatically on startup if they do not exist.
- Webcam access is required for scanning and enrollment features.

## License
This project is licensed under the MIT License.

## Contact
For questions or support, please contact the project maintainer.
