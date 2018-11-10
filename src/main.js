
import * as tf from '@tensorflow/tfjs'
import { Webcam } from './webcam'
// import * from 'readgyros';

import * as $ from 'jquery'

import * as apclust from 'affinity-propagation'

function log (l) {
  // $('#log').append(l + '<br>')
  console.log(l)
}
var running = false
async function runmodel () {
  log('getting image')
  var output = tf.tidy(() => {
    var image = window.webcam.capture()
    window.image = image
    // log("image " + image.shape);

    log('running model')
    var output = window.model.predict(image)

    output = tf.reshape(output, [128, 128, 2])

    window.output = output

    output = tf.concat([output, tf.add(1, tf.mul(-0.001, output))], 2)

    return output.add(-0.5).mul(3000).clipByValue(0, 1)
  })
  tf.toPixels(output, document.getElementById('segmentation'))
  //log('done')

  window.output_array = await output.slice([0, 0, 1], [128, 128, 1]).data()

  getlines(window.output_array)
  //running = false;
  if (running) {
    setTimeout(runmodel, 30)
  }
}

function intersection(line1, line2){
    //Finds the intersection of two lines given in Hesse normal form.

//
    //Returns closest integer pixel locations.
    //See https://stackoverflow.com/a/383527/5087436
/*
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    
    return [[x0, y0]]*/
}
function drawLines(lines, mat) {
	for (let i = 0; i < lines.rows; ++i) {
    let rho = lines.data32F[i * 2];
    let theta = lines.data32F[i * 2 + 1];
    let a = Math.cos(theta);
    let b = Math.sin(theta);
    let x0 = a * rho;
    let y0 = b * rho;
    let startPoint = {x: x0 - 1000 * b, y: y0 + 1000 * a};
    let endPoint = {x: x0 + 1000 * b, y: y0 - 1000 * a};
    cv.line(mat, startPoint, endPoint, [125, 0, 0, 255]);
}
}
function getlines (array) {
  window.mat = cv.matFromArray(128, 128, cv.CV_32F, array)
  var gr = new cv.Mat()
  mat.convertTo(mat, cv.CV_8UC1)
  cv.threshold(mat, gr, .5, 256, cv.THRESH_BINARY_INV)
  window.gr = gr
  var skele = cv.Mat.zeros(128, 128, cv.CV_8UC1)
  var temp = cv.Mat.zeros(128, 128, cv.CV_8UC1)
  var elem = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(3, 3))
  var i
  for (i = 0; i < 30; i++) {
    cv.morphologyEx(gr, temp, cv.MORPH_OPEN, elem)
    cv.bitwise_not(temp, temp)
    cv.bitwise_and(gr, temp, temp)
    cv.bitwise_or(skele, temp, skele)
    cv.erode(gr, gr, elem)
  }
  
  cv.imshow('skeleton', skele)
  mat.delete(); gr.delete(); temp.delete()
  let dst = cv.Mat.zeros(skele.rows, skele.cols, cv.CV_8UC3);
  let lines = new cv.Mat();

  cv.HoughLines(skele, lines, 1, Math.PI / 180,
              27, 0, 0, 0, Math.PI);
  drawLines(lines, dst)
  if (lines.rows < 2){
    return
  }
  
  var lineDistMat = []
  for (let i = 0; i < lines.rows; ++i){
    lineDistMat.push([])
    for (let j = 0; j < lines.rows; ++j) {
      var rho1 = lines.data32F[i * 2];
      var theta1 = lines.data32F[i * 2 + 1];

      let rho2 = lines.data32F[j * 2];
      let theta2 = lines.data32F[j * 2 + 1];

      let distance1 = Math.sqrt((rho1 - rho2) * (rho1 - rho2) / (100 * 100) + (theta1 - theta2) * (theta1 - theta2))
      rho1 = rho1 * -1
      theta1 = theta1 - Math.PI
      let distance2 = Math.sqrt((rho1 - rho2) * (rho1 - rho2) / (100 * 100) + (theta1 - theta2) * (theta1 - theta2))
      //console.log(distance1, distance2, rho1, theta1)
      lineDistMat[i].push(Math.max(-distance1, -distance2))
  	}
  }
  //console.table(lineDistMat)
  var cluster_result = apclust.getClusters(lineDistMat, {preference:window.preference, damping:.5})
  //console.log(cluster_result)
  //console.table(lineDistMat)
  for (let j = 0; j < cluster_result.exemplars.length; ++j) {
  	let i = cluster_result.exemplars[j]
  	console.log(i)
    let rho = lines.data32F[i * 2];
    let theta = lines.data32F[i * 2 + 1];
    let a = Math.cos(theta);
    let b = Math.sin(theta);
    let x0 = a * rho;
    let y0 = b * rho;
    let startPoint = {x: x0 - 1000 * b, y: y0 + 1000 * a};
    let endPoint = {x: x0 + 1000 * b, y: y0 - 1000 * a};
    cv.line(dst, startPoint, endPoint, [255, 255, 0, 255]);
  }

cv.imshow('lines', dst);
  dst.delete();

}
window.preference = -.6;
var sessionid = Math.round(90000 * Math.random())
var recording = false
function submitPhoto () {
  var context = document.getElementById("cameraframe").getContext("2d")

  context.drawImage(window.webcam.webcamElement, 0, 0, 512, 512)
  $.ajax({
  	type:"POST",
  	url:"/imageupload/" + (Math.random() + sessionid), 
  	data: {
  		image: document.getElementById("cameraframe").toDataURL('image/png')
  	}

  })
  if (recording){
    setTimeout(submitPhoto, 300)
  }
}


async function init () {
  //screen.lockOrientationUniversal = screen.lockOrientation || screen.mozLockOrientation || screen.msLockOrientation;
  //screen.lockOrientationUniversal("portrait-primary")
  log('document.load happened')

  window.model = await tf.loadModel('/webgrid/static/tfjs_dir/model.json')

  log('model loaded')

  window.webcam = new Webcam(document.getElementById('player'))

  await window.webcam.setup()

  window.webcam.webcamElement.width = 128
  window.webcam.webcamElement.height = 128

  log('Webcam setup')

  window.events = []
  /*
  $('#save').click(function () {
    $.ajax({
      type: 'POST',
      contentType: 'application/json; charset=utf-8',
      url: '/upload',
      data: JSON.stringify(window.events),
      datatype: 'json'
    })
    // events = []
  })
  */
  $('#runmodel').click(function () {
    running = !running
    $("#runmodel").text (running ? "Stop Model" : "Run Model")
    if (running) {

      runmodel()
    }
  })

  $('#runonce').click(function () {
    runmodel()
  })

  $('#stopmodel').click(function () {
    running = false
  })
  $('#submit').click(function () {
  	recording = !recording
    window.webcam.webcamElement.width = recording ? 512 : 128
    window.webcam.webcamElement.height = window.webcam.webcamElement.width

    $('#submit').text( recording ? "Stop Recording" : "Record and Upload")
  	if (recording ){
      submitPhoto()
  	}
  });

  if (window.DeviceOrientationEvent) {
    log('registering event')
    window.addEventListener('deviceorientation', function (evt) {
      window.events.push({
        time: Date.now() / 1000,
        alpha: evt.alpha,
        beta: evt.beta,
        gamma: evt.gamma
      })
      $('#alpha').text(evt.alpha)
      $('#beta').text(evt.beta)
      $('#gamma').text(evt.gamma)
      $('#webkit').text(evt.webkitCompassHeading)
    })
  } else {
    $('log').append('No DeviceOrientationEvent')
  }

  /* const player = document.getElementById('player')

      const constraints = {
        video: true,
      };

      navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
          player.srcObject = stream;
        });
       */
}

function initAwait () {
  init()
}
window.onload = initAwait
