import urllib.request, json

# 1. Create session
req = urllib.request.Request('http://127.0.0.1:8000/api/sessions', data=b'', headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(req) as resp:
    sess = json.loads(resp.read().decode('utf-8'))['session_id']
print("Session:", sess)

# 2. Ingest image
boundary = '----Boundary12345'
with open('frontend/public/sample_imagery/pair_00000_optical.jpg', 'rb') as f:
    img_bytes = f.read()

body = (
    f'--{boundary}\r\n'.encode('latin-1') +
    b'Content-Disposition: form-data; name="session_id"\r\n\r\n' +
    sess.encode('utf-8') +
    f'\r\n--{boundary}\r\n'.encode('latin-1') +
    b'Content-Disposition: form-data; name="files"; filename="optical.jpg"\r\n' +
    b'Content-Type: image/jpeg\r\n\r\n' +
    img_bytes +
    f'\r\n--{boundary}--\r\n'.encode('latin-1')
)

req2 = urllib.request.Request('http://127.0.0.1:8000/api/ingest', data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}, method='POST')
with urllib.request.urlopen(req2) as resp:
    imgs = json.loads(resp.read().decode('utf-8'))['images']
    img_id = imgs[0]['image_id']
print("Ingested Image ID:", img_id)

queries = [
    "Is there an airport or runway here?",
    "What buildings or industrial facilities are located in this area?",
    "Are there roads or transport corridors?",
    "Is there any water body or lake?",
    "How much vegetation or agriculture is present?",
    "What are the main land cover categories?"
]

for q in queries:
    q_body = json.dumps({'session_id': sess, 'query': q, 'image_ids': [img_id]}).encode('utf-8')
    req3 = urllib.request.Request('http://127.0.0.1:8000/api/query', data=q_body, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req3) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print(f"\n[QUERY]: {q}")
        print(f"[TASK]: {res['task']}")
        print(f"[ANSWER]: {res.get('answer')}")
