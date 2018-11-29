import * as $ from 'jquery'

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
  
  
  gridslam.init()
  $('#save').click(function () {
    $.ajax({
      type: 'POST',
      contentType: 'application/json; charset=utf-8',
      url: '/uploadtrack',
      data: JSON.stringify({"imu_events": window.events, "transforms":window.transforms}),
      datatype: 'json'
    })
    // events = []
  })
  
  

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
    window.webcam.webcamElement.width = recording ? 512 : 128
    window.webcam.webcamElement.height = window.webcam.webcamElement.width

    $('#submit').text( recording ? "Stop Recording" : "Record and Upload")
  	if (recording ){
      submitPhoto()
  	}
  });

  
  function checkLoadingDone() {
    if(cv.Mat && window.webcam){
      $(".loading-fade").hide()
      $(".loading-msg").hide()
      clickrun()
    }
    else setTimeout(checkLoadingDone, 130)
  }
  setTimeout(checkLoadingDone, 500)
  
}

function initAwait () {
  init()
}
window.onload = initAwait
