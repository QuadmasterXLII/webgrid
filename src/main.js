
import * as tf from '@tensorflow/tfjs'
import { Webcam } from './webcam'
// import * from 'readgyros';

import * as $ from 'jquery'

function log (l) {
  // $('#log').append(l + '<br>')
  console.log(l)
}
var running = false;
function runmodel () {
  log('getting image')
  var image = window.webcam.capture()
  window.image = image
  // log("image " + image.shape);

  log('running model')
  var output = window.model.predict(image)
  output = tf.reshape(output, [128, 128, 2])

  window.output = output

  output = tf.concat([output, tf.add(1, tf.mul(-0.001, output))], 2)

  output = output.add(-.5).mul(3000).clipByValue(0, 1)
  tf.toPixels(output, document.getElementById('segmentation'))
  log('done')
  if (running){
     setTimeout(runmodel, 100)
  }

}

async function run () {
  log('document.load happened')

  window.model = await tf.loadModel('tfjs_dir/model.json')

  log('model loaded')

  window.webcam = new Webcam(document.getElementById('player'))

  await window.webcam.setup()

  window.webcam.webcamElement.width = 128
  window.webcam.webcamElement.height = 128

  log('Webcam setup')

  window.events = []

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

  $('#runmodel').click(function () {
  	running = true;
    runmodel()
  })

  $('#stopmodel').click(function () {
    running = false;
  })

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

function runAwait () {
  run()
}
window.onload = runAwait
