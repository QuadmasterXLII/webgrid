
import * as tf from '@tensorflow/tfjs';
import {Webcam} from './webcam';
//import * from 'readgyros';

import * as $ from 'jquery';
console.log("running");

function log(l){
	$("#log").append(l + "<br>");

}



async function run() {
    console.log("document.load happened");
    log("document.load happened")

    

    window.model = await tf.loadModel('/static/tfjs_dir/model.json')
        
    window.model = model;
    log("model loaded")

    window.webcam = new Webcam(document.getElementById("player"));
    
    await webcam.setup();


    log("Webcam setup");

    window.events = [];

	$("#save").click(function () {
			    $.ajax({
			        type: "POST",
			        contentType: "application/json; charset=utf-8",
			        url: "/upload",
			        data: JSON.stringify(events),
			        datatype: "json"
			    });
			    //events = []
			});

	if (window.DeviceOrientationEvent) {
	    $("#log").append("registering event")
	    window.addEventListener('deviceorientation', function (evt) {
	        window.events.push({
	            time: Date.now() / 1000,
	            alpha: evt.alpha,
	            beta: evt.beta,
	            gamma: evt.gamma
	        });
	        $("#alpha").text(evt.alpha);
	        $("#beta").text(evt.beta);
	        $("#gamma").text(evt.gamma);
	        $("#webkit").text(evt.webkitCompassHeading);
	        
	    })
	} else {
	    $('log').append("No DeviceOrientationEvent");
	}

	const player = document.getElementById('player');
      /*
	  const constraints = {
	    video: true,
	  };

	  navigator.mediaDevices.getUserMedia(constraints)
	    .then((stream) => {
	      player.srcObject = stream;
	    });
	   */
	
}
 
function run_await() {
	run();
}
window.onload = run_await;
