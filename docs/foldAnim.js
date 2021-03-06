(function (W, D) {

var SZ = 32;
var GAP = 32;
// var cUNetFold = D.getElementById("UNetFold");
// var ctxUNetFold = cUNetFold.getContext("2d");

function drawArrow(fromx, fromy, tox, toy, c, s) {
  c.fillStyle = s || 'Black';
  c.strokeStyle = s || 'Black';
  // HACK
  var A = 10;
  var headlen = 10;
  var angle = Math.atan2(toy-fromy, tox-fromx);
  c.beginPath();
  c.moveTo(fromx, fromy);
  c.lineTo(tox, toy);
  c.moveTo(tox, toy);
  // c.moveTo(tox, toy);
  c.lineTo(tox-headlen*Math.cos(angle-Math.PI/A),toy-headlen*Math.sin(angle-Math.PI/A));
  // c.moveTo(tox, toy);
  c.lineTo(tox-headlen*Math.cos(angle+Math.PI/A),toy-headlen*Math.sin(angle+Math.PI/A));
  c.lineTo(tox, toy);
  c.stroke();
  c.fill();
}


function drawBox(x, y, skew, c) {
  s = SZ * skew;
  c.beginPath();
  c.moveTo(x - SZ / 2 + s, y - SZ / 2 + s/2);
  c.lineTo(x + SZ / 2 + s, y - SZ / 2 + s/2);
  c.lineTo(x + SZ / 2 - s, y + SZ / 2 - s/2);
  c.lineTo(x - SZ / 2 - s, y + SZ / 2 - s/2);
  c.lineTo(x - SZ / 2 + s, y - SZ / 2 + s/2);
  c.fill();
  c.stroke();
}

function p2skew(progress) {
  return progress / 4;
}

var YOFF = 400;
function linp(p, a, b) {
  return a + p * (b - a);
}


function drawFoldF0(p, c) {
  for (var i = 0; i < 4; i++) {
    c.fillStyle = "MediumSlateBlue";
    drawBox(
      linp(p, 50, 190 + (SZ + GAP) * (2 * i)),
      linp(p, 50 + SZ * i, YOFF - 340),
      p2skew(p), c);
  }
  for (var i = 0; i < 4; i++) {
    c.fillStyle = "LightSkyBlue";
    drawBox(
      linp(p, 120, 190 + (SZ + GAP) * (2 * i)),
      linp(p, 50 + SZ * i, YOFF - 340 + (SZ + 8)),
      p2skew(p), c);
  }
}

function drawFoldF1(p, c) {
  for (var i = 0; i < 2; i++) {
    c.fillStyle = "MediumSlateBlue";
    drawBox(
      linp(p, 120, 190 + (SZ + GAP) * (2 * i + 2)),
      linp(p, 216 + SZ * i, YOFF - 220),
      p2skew(p), c);
  }
  for (var i = 0; i < 2; i++) {
    c.fillStyle = "LightSkyBlue";
    drawBox(
      linp(p, 190, 190 + (SZ + GAP) * (2 * i + 2)),
      linp(p, 216 + SZ * i, YOFF - 220 + (SZ + 8)),
      p2skew(p), c);
  }
}

function drawFoldF2(p, c) {
  for (var i = 0; i < 1; i++) {
    c.fillStyle = "MediumSlateBlue";
    drawBox(
      linp(p, 190, 190 + (SZ + GAP) * (2 * i + 3)),
      linp(p, 320 + SZ * i, YOFF - 100),
      p2skew(p), c);
  }
  for (var i = 0; i < 1; i++) {
    c.fillStyle = "LightSkyBlue";
    drawBox(
      linp(p, 260, 190 + (SZ + GAP) * (2 * i + 3)),
      linp(p, 320 + SZ * i, YOFF - 100 + (SZ + 8)),
      p2skew(p), c);
  }
}

function drawFoldE2(p, c) {
  for (var i = 0; i < 1; i++) {
    c.fillStyle = "Pink";
    drawBox(
      linp(p, 330, 190 + (SZ + GAP) * (2 * i + 4)),
      linp(p, 320 + SZ * i, YOFF - 100 + (SZ + 8)),
      p2skew(p), c);
  }
  for (var i = 0; i < 1; i++) {
    c.fillStyle = "Tomato";
    drawBox(
      linp(p, 400, 190 + (SZ + GAP) * (2 * i + 4)),
      linp(p, 320 + SZ * i, YOFF - 100),
      p2skew(p), c);
  }
}

function drawFoldE1(p, c) {
  for (var i = 0; i < 2; i++) {
    c.fillStyle = "Pink";
    drawBox(
      linp(p, 400, 190 + (SZ + GAP) * (2 * i + 3)),
      linp(p, 216 + SZ * i, YOFF - 220 + (SZ + 8)),
      p2skew(p), c);
  }
  for (var i = 0; i < 2; i++) {
    c.fillStyle = "Tomato";
    drawBox(
      linp(p, 470, 190 + (SZ + GAP) * (2 * i + 3)),
      linp(p, 216 + SZ * i, YOFF - 220),
      p2skew(p), c);
  }
}

function drawFoldE0(p, c) {
  for (var i = 0; i < 4; i++) {
    c.fillStyle = "Pink";
    drawBox(
      linp(p, 470, 190 + (SZ + GAP) * (2 * i + 1)),
      linp(p, 50 + SZ * i, YOFF - 340 + (SZ + 8)),
      p2skew(p), c);
  }
  for (var i = 0; i < 4; i++) {
    c.fillStyle = "Tomato";
    drawBox(
      linp(p, 540, 190 + (SZ + GAP) * (2 * i + 1)),
      linp(p, 50 + SZ * i, YOFF - 340),
      p2skew(p), c);
  }
}


function convArrows(p, c) {
  // F conv

  for (var i = 0; i < 4; i++) {
    drawArrow(
      linp(p, 70, 190 + (SZ + GAP) * (2 * i)),
      linp(p, 98, YOFF - 328),
      linp(p, 98, 190 + (SZ + GAP) * (2 * i)),
      linp(p, 98, YOFF - 312),
      c);
  }
  for (var i = 0; i < 2; i++) {
    drawArrow(
      linp(p, 140, 190 + (SZ + GAP) * (2 * i + 2)),
      linp(p, 232, YOFF - 208),
      linp(p, 168, 190 + (SZ + GAP) * (2 * i + 2)),
      linp(p, 232, YOFF - 192),
      c);
  }
  drawArrow(
    linp(p, 210, 190 + (SZ + GAP) * (3)),
    linp(p, 320, YOFF - 88),
    linp(p, 238, 190 + (SZ + GAP) * (3)),
    linp(p, 320, YOFF - 72),
    c);

  // E conv
  for (var i = 0; i < 4; i++) {
    drawArrow(
      linp(p, 490, 190 + (SZ + GAP) * (2 * i + 1)),
      linp(p, 98, YOFF - 312),
      linp(p, 518, 190 + (SZ + GAP) * (2 * i + 1)),
      linp(p, 98, YOFF - 328),
      c);
  }
  for (var i = 0; i < 2; i++) {
    drawArrow(
      linp(p, 420, 190 + (SZ + GAP) * (2 * i + 3)),
      linp(p, 232, YOFF - 192),
      linp(p, 448, 190 + (SZ + GAP) * (2 * i + 3)),
      linp(p, 232, YOFF - 208),
      c);
  }
  drawArrow(
      linp(p, 350, 190 + (SZ + GAP) * (4)),
      linp(p, 320, YOFF - 72),
      linp(p, 378, 190 + (SZ + GAP) * (4)),
      linp(p, 320, YOFF - 88),
      c);

  // Downsample
  for (var i = 0; i < 4; i++) {
    drawArrow(
      linp(p, 120, 190 + (SZ + GAP) * (2 * i + 0)),
      linp(p, 168, YOFF - 316 + SZ),
      linp(p, 120, 190 + (SZ + GAP) * (2 * (i/2|0) + 2)),
      linp(p, 194, YOFF - 204 - SZ),
      c, "MediumSlateBlue");
  }
  for (var i = 0; i < 2; i++) {
    drawArrow(
      linp(p, 120 + 70, 190 + (SZ + GAP) * (2 * i + 2)),
      linp(p, 174 + 3 * SZ, YOFF - 196 + SZ),
      linp(p, 120 + 70, 190 + (SZ + GAP) * (2 * (i/2|0) + 3)),
      linp(p, 200 + 3 * SZ, YOFF - 84 - SZ),
      c, "MediumSlateBlue");
  }
  // Upsample
  for (var i = 0; i < 2; i++) {
    drawArrow(
      linp(p, 120 + 4 * 70, 190 + (SZ + GAP) * (2 * (i/2|0) + 4)),
      linp(p, 200 + 3 * SZ, YOFF - 84 - SZ),
      linp(p, 120 + 4 * 70, 190 + (SZ + GAP) * (2 * i + 3)),
      linp(p, 174 + 3 * SZ, YOFF - 196 + SZ),
      c, "Tomato");
  }
  for (var i = 0; i < 4; i++) {
    drawArrow(
      linp(p, 120 + 5 * 70, 190 + (SZ + GAP) * (2 * (i/2|0) + 3)),
      linp(p, 194, YOFF - 204 - SZ),
      linp(p, 120 + 5 * 70, 190 + (SZ + GAP) * (2 * i + 1)),
      linp(p, 168, YOFF - 316 + SZ),
      c, "Tomato");
  }

  // Copies (TODO)
  for (var i = 0; i < 4; i++) {
    drawArrow(
      linp(p, 140 + 10, 190 + (SZ + GAP) * (2 * i) + 20),
      linp(p, 98, YOFF - 340 + (SZ + 8)),
      linp(p, 448 - 10, 190 + (SZ + GAP) * (2 * i) + 20 + 24),
      linp(p, 98, YOFF - 340 + (SZ + 8)),
      c, "Grey");
  }
  for (var i = 0; i < 2; i++) {
    drawArrow(
      linp(p, 210 + 10, 190 + (SZ + GAP) * (2 * i + 2) + 20),
      linp(p, 232, YOFF - 220 + (SZ + 8)),
      linp(p, 378 - 10, 190 + (SZ + GAP) * (2 * i + 2) + 20 + 24),
      linp(p, 232, YOFF - 220 + (SZ + 8)),
      c, "Grey");
  }
  drawArrow(
    linp(p, 280, 190 + (SZ + GAP) * (3) + 20),
    linp(p, 320, YOFF - 100 + (SZ + 8)),
    linp(p, 308, 190 + (SZ + GAP) * (3) + 20 + 24),
    linp(p, 320, YOFF - 100 + (SZ + 8)),
    c, "Grey");
}


function drawFoldAnim(progress) {
  cElt = D.getElementById("UNetFold");
  c = cElt.getContext("2d");
  c.clearRect(0, 0, cElt.width, cElt.height);
  c.lineWidth = 2;
  c.strokeStyle = "black";
  drawFoldF0(progress, c);
  drawFoldF1(progress, c);
  drawFoldF2(progress, c);
  drawFoldE2(progress, c);
  drawFoldE1(progress, c);
  drawFoldE0(progress, c);
  c.lineWidth = 3;
  c.fillStyle = "black";
  convArrows(progress, c);
}

///// Exported
W.attachFoldAnimation = function(elt) {
  elt.addEventListener('input', function(e) { drawFoldAnim(elt.value); })
  drawFoldAnim(elt.value);
}



})(window, document);
