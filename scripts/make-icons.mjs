// Renders the icon.svg design (a warm glow in the dark) to PNG for the sizes
// that cannot be an SVG. Run: node scripts/make-icons.mjs
import { deflateSync } from 'node:zlib'
import { writeFileSync } from 'node:fs'

const AMBER = [255, 210, 143]
const ORANGE = [255, 84, 10]
const CORE = [255, 217, 160]
const SAMPLES = 3

function halo(t) {
  // Matches the SVG stops: amber at the centre, orange, then nothing.
  if (t >= 1) return [ORANGE, 0]
  if (t < 0.3) {
    const k = t / 0.3
    return [AMBER.map((c, i) => c + (ORANGE[i] - c) * k), 0.95 + (0.7 - 0.95) * k]
  }
  const k = (t - 0.3) / 0.7
  return [ORANGE, 0.7 * (1 - k)]
}

function pixel(x, y, size) {
  const c = size / 2
  const haloR = (size / 64) * 22
  const coreR = (size / 64) * 4.5
  let r = 0
  let g = 0
  let b = 0
  for (let sy = 0; sy < SAMPLES; sy++) {
    for (let sx = 0; sx < SAMPLES; sx++) {
      const dx = x + (sx + 0.5) / SAMPLES - c
      const dy = y + (sy + 0.5) / SAMPLES - c
      const d = Math.hypot(dx, dy)
      let [color, alpha] = halo(d / haloR)
      let out = color.map((v) => v * alpha)
      if (d <= coreR) {
        out = CORE
      }
      r += out[0]
      g += out[1]
      b += out[2]
    }
  }
  const n = SAMPLES * SAMPLES
  return [Math.round(r / n), Math.round(g / n), Math.round(b / n)]
}

const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})

function crc32(buf) {
  let c = 0xffffffff
  for (const byte of buf) c = crcTable[(c ^ byte) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

function chunk(type, data) {
  const length = Buffer.alloc(4)
  length.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body))
  return Buffer.concat([length, body, crc])
}

function png(size) {
  const raw = Buffer.alloc(size * (size * 3 + 1))
  let p = 0
  for (let y = 0; y < size; y++) {
    raw[p++] = 0
    for (let x = 0; x < size; x++) {
      const [r, g, b] = pixel(x, y, size)
      raw[p++] = r
      raw[p++] = g
      raw[p++] = b
    }
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(size, 0)
  ihdr.writeUInt32BE(size, 4)
  ihdr[8] = 8
  ihdr[9] = 2
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

// An .ico is just a header wrapped around a PNG, which every current browser reads.
function ico(size) {
  const image = png(size)
  const header = Buffer.alloc(22)
  header.writeUInt16LE(0, 0)
  header.writeUInt16LE(1, 2)
  header.writeUInt16LE(1, 4)
  header[6] = size
  header[7] = size
  header.writeUInt16LE(1, 10)
  header.writeUInt16LE(32, 12)
  header.writeUInt32BE(0, 14)
  header.writeUInt32LE(image.length, 14)
  header.writeUInt32LE(22, 18)
  return Buffer.concat([header, image])
}

for (const [path, size] of [
  ['app/apple-icon.png', 180],
  ['public/icon-192.png', 192],
  ['public/icon-512.png', 512],
]) {
  writeFileSync(path, png(size))
  console.log(`${path} ${size}x${size}`)
}

writeFileSync('app/favicon.ico', ico(32))
console.log('app/favicon.ico 32x32')
