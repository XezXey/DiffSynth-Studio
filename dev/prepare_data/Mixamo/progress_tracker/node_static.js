const http = require('http');
const fs   = require('fs');
const path = require('path');
const crypto = require('crypto');

const RENDER_ROOT = process.env.RENDER_ROOT || '/data/mint/Motion_Dataset/Mixamo/render_fbx';
const PORT        = parseInt(process.env.STATIC_PORT || '5001');
const ALLOWED_EXT = new Set(['.png', '.jpg', '.jpeg', '.webp', '.exr']);

const MIME = {
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.exr':  'image/x-exr',
};

const server = http.createServer((req, res) => {
  // Strip query string
  const urlPath = req.url.split('?')[0];

  // Security: block path traversal
  const rel = path.normalize(urlPath).replace(/^(\.\.[\/\\])+/, '');
  const ext = path.extname(rel).toLowerCase();

  if (!ALLOWED_EXT.has(ext)) {
    res.writeHead(403); res.end(); return;
  }

  const fullPath = path.join(RENDER_ROOT, rel);

  fs.stat(fullPath, (err, stat) => {
    if (err || !stat.isFile()) {
      res.writeHead(404); res.end(); return;
    }

    const size  = stat.size;
    const mtime = stat.mtimeMs;

    // ETag
    const etag = '"' + crypto.createHash('md5')
      .update(`${fullPath}:${mtime}:${size}`).digest('hex').slice(0, 16) + '"';

    if (req.headers['if-none-match'] === etag) {
      res.writeHead(304); res.end(); return;
    }

    // Range support
    const rangeHeader = req.headers['range'];
    let start = 0, end = size - 1, status = 200;

    if (rangeHeader && rangeHeader.startsWith('bytes=')) {
      const parts = rangeHeader.slice(6).split('-');
      start  = parts[0] ? parseInt(parts[0]) : 0;
      end    = parts[1] ? parseInt(parts[1]) : size - 1;
      end    = Math.min(end, size - 1);
      status = 206;
    }

    const chunkSize = end - start + 1;
    const headers = {
      'Content-Type':                  MIME[ext] || 'application/octet-stream',
      'Content-Length':                chunkSize,
      'ETag':                          etag,
      'Accept-Ranges':                 'bytes',
      'Cache-Control':                 'public, max-age=3600, immutable',
      'Last-Modified':                 new Date(mtime).toUTCString(),
      'Access-Control-Allow-Origin':   '*',
    };
    if (status === 206) {
      headers['Content-Range'] = `bytes ${start}-${end}/${size}`;
    }

    res.writeHead(status, headers);

    if (req.method === 'HEAD') { res.end(); return; }

    // Stream the file — Node uses libuv which calls sendfile on Linux
    const stream = fs.createReadStream(fullPath, { start, end });
    stream.pipe(res);
    stream.on('error', () => { res.destroy(); });
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[node_static] Serving ${RENDER_ROOT} on port ${PORT}`);
});
