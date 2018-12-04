import * as $ from 'jquery'
import * as kalman from "./kalman"
import * as utils from "./utils"

import * as gridslam from "./gridslam"


var sessionid = Math.round(90000 * Math.random())
var recording = false
function submitPhoto () {
  var context = document.getElementById("cameraframe").getContext("2d")

  context.drawImage(window.webcam.webcamElement, 0, 0, 512, 512)
  $.ajax({
  	type:"POST",
  	url:"http://ec2-18-188-175-17.us-east-2.compute.amazonaws.com:4000/imageupload/" + (Math.random() + sessionid), 
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

  
  
  await gridslam.init()


  kalman.setCallback ( (x) => {
      
      
      var input_shape = gridslam.input_shape

      window.trackingctx = document.getElementById("tracking").getContext("2d")

      window.trackingctx.fillStyle = "rgba(255,255,255,1)"
      window.trackingctx.fillRect(0, 0, input_shape, input_shape)
      window.trackingctx.fillStyle = "rgba(255,0,0,1)"
      window.trackingctx.fillRect(input_shape * utils.mod1(x[0].data32F[0]), input_shape - input_shape * utils.mod1(x[1].data32F[0]), 4, 4)


  })
  $('#save').click(function () {
    $.ajax({
      type: 'POST',
      contentType: 'application/json; charset=utf-8',
      url: '/uploadtrack',
      data: JSON.stringify({
        "orientation_events": gridslam.orientation_events,
        "acceleration_events":gridslam.acceleration_events,
        "transforms":window.transforms,
        "imu_yaw2grid_yaw_delta": gridslam.imu_yaw2grid_yaw_delta
      }),
      datatype: 'json'
    })
    // events = []
  })
  $('#resetMap').click(gridslam.resetMap)
  

  function clickrun() {
    gridslam.setRunning(!gridslam.running)
    $("#runmodel").text (gridslam.running ? "Stop Model" : "Run Model")
    if (gridslam.running) {

      gridslam.runmodel()
    }
  }
  $('#runmodel').click(clickrun)

  $('#runonce').click(function () {
    gridslam.runmodel()
  })

  $('#stopmodel').click(function () {
    gridslam.running = false
  })
  $('#submit').click(function () {
  	recording = !recording
    window.webcam.webcamElement.width = recording ? 512 : gridslam.input_shape
    window.webcam.webcamElement.height = window.webcam.webcamElement.width

    $('#submit').text( recording ? "Stop Recording" : "Record and Upload")
  	if (recording ){
      submitPhoto()
  	}
  });

  
  function checkLoadingDone() {
    try {
      if(cv.Mat && window.webcam){
        $(".loading-fade").hide()
        $(".loading-msg").hide()
        kalman.init()
        clickrun()
      } else {
        setTimeout(checkLoadingDone, 130)
      }
    }
    catch (err) {
      setTimeout(checkLoadingDone, 130)
    }
  }
  setTimeout(checkLoadingDone, 50)
  
}

function initAwait () {
  init()
}
cv['onRuntimeInitialized'] = initAwait
