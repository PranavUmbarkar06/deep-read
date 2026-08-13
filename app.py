from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import werkzeug
from graph import build_graph
from pypdf import PdfReader

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'downloaded_papers')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Compile LangGraph workflow engine
print("[Deep Read] Compiling Agentic Workflow Graph...")
graph = build_graph()

@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join('frontend', path)):
        return send_from_directory('frontend', path)
    return send_from_directory('frontend', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No file provided in request'}), 400
    
    files = request.files.getlist('files') or request.files.getlist('file')
    uploaded_info = []

    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = werkzeug.utils.secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            
            try:
                reader = PdfReader(save_path)
                page_count = len(reader.pages)
            except Exception:
                page_count = 0

            uploaded_info.append({
                'name': filename,
                'path': save_path,
                'size': os.path.getsize(save_path),
                'pages': page_count
            })

    return jsonify({'uploaded': uploaded_info})

@app.route('/api/documents', methods=['GET'])
def get_documents():
    docs = []
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if f.lower().endswith('.pdf'):
                full_path = os.path.join(UPLOAD_FOLDER, f)
                try:
                    reader = PdfReader(full_path)
                    pages = len(reader.pages)
                except Exception:
                    pages = 0
                docs.append({
                    'name': f,
                    'path': full_path,
                    'size': os.path.getsize(full_path),
                    'pages': pages
                })
    return jsonify({'documents': docs})

@app.route('/api/documents/<filename>', methods=['DELETE'])
def delete_document(filename):
    secure_name = werkzeug.utils.secure_filename(filename)
    target = os.path.join(UPLOAD_FOLDER, secure_name)
    if os.path.exists(target):
        try:
            os.remove(target)
            return jsonify({'success': True, 'message': f'Deleted {secure_name}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    query = data.get('query', '')
    pdf_paths = data.get('pdf_paths', [])
    thread_id = data.get('thread_id', 'default-session')  # or generate a uuid per browser session

    if not query:
        return jsonify({'error': 'Query string is required'}), 400

    try:
        state_input = {
            "query": query,
            "pdf_paths": pdf_paths
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(state_input, config)
        interrupt_payload = result.get("__interrupt__")
        interrupt_message = None
        if interrupt_payload:
            interrupt_message = interrupt_payload[0].value if hasattr(interrupt_payload[0], "value") else str(interrupt_payload[0])
        
        return jsonify({
            'intent': result.get('intent'),
            'router_reasoning': result.get('router_reasoning'),
            'final_message': interrupt_message or result.get('final_message'),
            'papers': result.get('papers', []),
            'candidate_titles': result.get('candidate_titles', []),
            'is_compatible': result.get('is_compatible'),
            'compatibility_reason': result.get('compatibility_reason'),
            'comparison_analysis': result.get('comparison_analysis'),
            'comparison_summary': result.get('comparison_summary'),
            'validate_result': result.get('validate_result')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("[Deep Read] Starting Flask Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
