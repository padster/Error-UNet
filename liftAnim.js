(function (W, D) {

var SZ = 32;
var GAP = 32;

var XOFF = 0;
var YOFF = 0;

function drawArrow(fromx, fromy, tox, toy, c, s) {
  fromx += XOFF; tox += XOFF;
  fromy += YOFF; toy += YOFF;
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
  x += XOFF;
  y += YOFF;
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

function linp(p, a, b) {
  return a + p * (b - a);
}


function drawOldF0(p, c) {
  for (var i = 0; i < 3; i++) {
    c.fillStyle = "MediumSlateBlue";
    drawBox(120 - i * SZ * 0 / 2, 350 - i * SZ * 1.5, 0.25, c);
  }
}

function drawNewF0(p, c) {
  for (var i = 0; i < 3; i++) {
    c.fillStyle = "MediumSlateBlue";
    drawBox(120 - i * SZ * 0 / 2, 300 - 5 * SZ - i * SZ * 1.5, 0.25, c);
  }
}

function drawNewE(p, c) {
  for (var i = 0; i < 3; i++) {
    c.fillStyle = "Tomato";
    drawBox(120 + 4 * SZ - i * SZ * 0 / 2, 286 - SZ - i * SZ * 1.5, 0.25, c);
  }
}

function drawOldE(p, c) {
  for (var i = 0; i < 3; i++) {
    c.fillStyle = "Tomato";
    drawBox(120 + 4 * SZ - i * SZ * 0 / 2, 336 + 4 * SZ - i * SZ * 1.5, 0.25, c);
  }
}


function convArrows(p, c) {
  // Old F
  drawArrow(120, 350 - SZ * 0.5 + 3, 120, 350 - SZ * 1.0 - 3, c);
  drawArrow(120, 350 - SZ * 2.0 + 3, 120, 350 - SZ * 2.5 - 3, c);
  // New F
  drawArrow(120, 300 - SZ * 5.5 + 3, 120, 300 - SZ * 6.0 - 3, c);
  drawArrow(120, 300 - SZ * 7.0 + 3, 120, 300 - SZ * 7.5 - 3, c);
  // Old E
  drawArrow(120 + 4 * SZ, 336 + SZ * 3.0 - 3, 120 + 4 * SZ, 336 + SZ * 3.5 + 3, c);
  drawArrow(120 + 4 * SZ, 336 + SZ * 1.5 - 3, 120 + 4 * SZ, 336 + SZ * 2.0 + 3, c);
  // New E
  drawArrow(120 + 4 * SZ, 286 - SZ * 2.0 - 3, 120 + 4 * SZ, 286 - SZ * 1.5 + 3, c);
  drawArrow(120 + 4 * SZ, 286 - SZ * 3.5 - 3, 120 + 4 * SZ, 286 - SZ * 3.0 + 3, c);
  // Old E -> Olf F
  drawArrow(120 + 3 * SZ, 336 + SZ * 4.2 + 3, 120 + 0.4 * SZ, 300 + SZ * 1.8 + 3, c);
  drawArrow(120 + 3 * SZ, 336 + SZ * 2.7 + 3, 120 + 0.4 * SZ, 300 + SZ * 0.3 + 3, c);
  drawArrow(120 + 3 * SZ, 336 + SZ * 1.2 + 3, 120 + 0.4 * SZ, 300 - SZ * 1.2 + 3, c);
  // Old F -> New E
  drawArrow(120 + SZ, 350 - SZ * 0.5 + 3, 120 + 3 * SZ, 286 - SZ * 0.5 + 3, c);
  drawArrow(120 + SZ, 350 - SZ * 2.0 + 3, 120 + 3 * SZ, 286 - SZ * 2.0 + 3, c);
  drawArrow(120 + SZ, 350 - SZ * 3.5 + 3, 120 + 3 * SZ, 286 - SZ * 3.5 + 3, c);
  // New E -> New F
  drawArrow(120 + 3 * SZ, 286 - SZ * 0.8 + 3, 120 + 0.4 * SZ, 350 - SZ * 6.2 + 3, c);
  drawArrow(120 + 3 * SZ, 286 - SZ * 2.3 + 3, 120 + 0.4 * SZ, 350 - SZ * 7.7 + 3, c);
  drawArrow(120 + 3 * SZ, 286 - SZ * 3.8 + 3, 120 + 0.4 * SZ, 350 - SZ * 9.2 + 3, c);

}


function drawLifAnim(progress) {
  cElt = D.getElementById("UNetFold");
  c = cElt.getContext("2d");
  c.clearRect(0, 0, cElt.width, cElt.height);
  c.lineWidth = 2;
  c.strokeStyle = "black";
  drawOldF0(progress, c);
  drawNewF0(progress, c);
  drawOldE(progress, c);
  drawNewE(progress, c);
  c.lineWidth = 3;
  c.fillStyle = "black";
  convArrows(progress, c);
}

///// Exported
W.attachLiftAnimation = function(elt) {
  elt.addEventListener('input', function(e) { drawLifAnim(elt.value); })
  drawLifAnim(elt.value);
}



})(window, document);
