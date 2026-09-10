const { exec } = require('child_process');
const http = require('http');

// Launch Edge with remote debugging
const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const userDataDir = 'C:\\Users\\vishu\\Documents\\satquery\\temp_edge_profile';
const edgeProcess = exec(`"${edgePath}" --headless=new --remote-debugging-port=9222 --user-data-dir="${userDataDir}" "http://127.0.0.1:5173"`);

setTimeout(() => {
  // Query http://127.0.0.1:9222/json to find target
  http.get('http://127.0.0.1:9222/json', (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      try {
        const targets = JSON.parse(data);
        console.log('Targets:', JSON.stringify(targets, null, 2));
      } catch (e) {
        console.error('Parse error:', e);
      }
      edgeProcess.kill();
    });
  }).on('error', (err) => {
    console.error('HTTP error:', err.message);
    edgeProcess.kill();
  });
}, 2000);
