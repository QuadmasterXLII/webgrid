import * as tf from '@tensorflow/tfjs'
import { Webcam } from './webcam'
import * as $ from 'jquery'
import * as apclust from 'affinity-propagation'

var clustermaker = require("clusters")
var fmin = require("fmin")



window.events = []
window.transforms = []
export async function init() {
  

  window.model = await tf.loadModel('/static/tfjs_dir/model.json')

  
  window.webcam = new Webcam(document.getElementById('player'))

  await window.webcam.setup()

  window.webcam.webcamElement.width = 128
  window.webcam.webcamElement.height = 128




  if (window.DeviceOrientationEvent) {
    
    window.addEventListener('deviceorientation', function (evt) {
      window.events.push({
        time: Date.now() / 1000,
        alpha: evt.alpha,
        beta: evt.beta,
        gamma: evt.gamma
      })
      /*
      $('#alpha').text(evt.alpha)
      $('#beta').text(evt.beta)
      $('#gamma').text(evt.gamma)
      $('#webkit').text(evt.webkitCompassHeading)
      */
    })
  } else {
    console.log('No DeviceOrientationEvent')
  }
} 
export var running = false
export function setRunning(r){
  running = r
}
export async function runmodel () {
 
  var imu_idx = window.events.length - 1 
  var output = tf.tidy(() => {
    var image = window.webcam.capture()
    window.image = image
 
    var output = window.model.predict(image)

    output = tf.reshape(output, [128, 128, 2])

    //window.output = output

    //output = tf.concat([output, tf.add(1, tf.mul(-0.001, output))], 2)

    return output.add(-0.5).mul(3000).clipByValue(0, 1).slice([0, 0, 1], [128, 128, 1])
  })
  //tf.toPixels(output, document.getElementById('segmentation'))
  //log('done')

  window.output_array = await output.data()
  setTimeout(()=> {
    getlines(window.output_array, imu_idx)
    //running = false;
    if (running) {
      setTimeout(runmodel, 5)
    }
  }, 0)
}

function intersection(line1, line2){
    //Finds the intersection of two lines given in Hesse normal form.
    //Returns closest integer pixel locations.
    //See https://stackoverflow.com/a/383527/5087436

    let rho1 = line1[0]
    let theta1 = line1[1]
    let rho2 = line2[0]
    let theta2 = line2[1]
    let A = cv.matFromArray(2, 2, cv.CV_32F, [
        Math.cos(theta1), Math.sin(theta1),
        Math.cos(theta2), Math.sin(theta2)
    ])
    let b = cv.matFromArray(2, 1, cv.CV_32F, [rho1, rho2])
    let res = new cv.Mat()
    cv.gemm(A.inv(cv.DECOMP_LU), b, 1, cv.matFromArray(2, 1, cv.CV_32F, [0, 0]), 0, res, 0)
    
    return res
}
function drawLines(lines, mat, color) {
    for (let i = 0; i < lines.rows; ++i) {
    let rho = lines.data32F[i * 3];
    let theta = lines.data32F[i * 3 + 1];
    let a = Math.cos(theta);
    let b = Math.sin(theta);
    let x0 = a * rho;
    let y0 = b * rho;
    let startPoint = {x: x0 - 1000 * b, y: y0 + 1000 * a};
    let endPoint = {x: x0 + 1000 * b, y: y0 - 1000 * a};
    cv.line(mat, startPoint, endPoint, color);
  }
}
function drawLinesJ(lines, mat, color) {
  for (let i = 0; i < lines.length; ++i) {
    let rho = lines[i][0];
    let theta = lines[i][1];
    let a = Math.cos(theta);
    let b = Math.sin(theta);
    let x0 = a * rho;
    let y0 = b * rho;
    let startPoint = {x: x0 - 1000 * b, y: y0 + 1000 * a};
    let endPoint = {x: x0 + 1000 * b, y: y0 - 1000 * a};
    cv.line(mat, startPoint, endPoint, color);
  }
}
function mat32FToArray(mat) {
  var res = []
  for (let i = 0; i < mat.rows; ++i) {
    res.push([])
    for (let j = 0; j < mat.cols; ++j) {
      res[i].push(mat.data32F[i * mat.cols + j])
    }
  }
  return res
}

function project_points(vector, world_points){
  var sin = Math.sin
  var cos = Math.cos
  var pi = Math.PI
  var height = 128
  var width = 128 / window.webcam.webcamElement.videoWidth * window.webcam.webcamElement.videoHeight

  var x = vector[0]
  var y = vector[1]
  var z = vector[2]
  var pitch =vector[3]
  var roll = vector[4]
  var yaw = vector[5]
  var focallength = vector[6]

  screen_points = []

  let x0  =  sin(roll)
  let x1  =  cos(pitch)
  let x2  =  world_z*x1
  let x3  =  x1*z
  let x4  =  cos(roll)
  let x5  =  cos(yaw)
  let x6  =  x4*x5
  let x7  =  sin(pitch)
  let x8  =  sin(yaw)
  let x9  =  x0*x8
  let x10  =  x6 + x7*x9
  let x11  =  x4*x8
  let x12  =  x0*x5
  let x13  =  -x11 + x12*x7
  let x14  =  x6*x7 + x9
  let x15  =  x11*x7 - x12
  let x17  =  x1*x8
  let x18  =  x1*x5
  let x_19 = x*x15 + x14*y + x2*x4 + x3*x4


  for(var i = 0; i < world_points.length; ++i) {
    var world_point = world_points[i]
    var world_x= world_point[0]
    var world_y = world_point[1]
    var world_z = world_point[2]

    
    let x16  =  focallength/(world_x*x15 + world_y*x14 + x_19)
    screen_points.push([height/2 - x16*(world_x*x10 + world_y*x13 + x*x10 + x0*x2 + x0*x3 + x13*y), width/2 - x16*(world_x*x17 + world_y*x18 - world_z*x7 + x*x17 + x18*y - x7*z)])

  }
  return screen_points
}


let error_scale = 100
window.cnt = 0
function make_error_params(vector, screen_points, world_points){
  vector = vector.slice()
  function error_restricted(vector2, fprime){
    window.cnt += 1;
    [[0, 0], [1, 1], [2, 2], [3, 5]].forEach((j)=>{
      vector[j[1]] = vector2[j[0]]
    })
    var sin = Math.sin
    var cos = Math.cos
    var pi = Math.PI
    var height = 128
    var width = 128 / window.webcam.webcamElement.videoWidth * window.webcam.webcamElement.videoHeight
    
    fprime = fprime || [0, 0, 0, 0]
    for(var j=0; j < 4; j++){
      fprime[j] = 0
    }
    //console.log(width, height)

    var x = vector[0]
    var y = vector[1]
    var z = vector[2]
    var pitch =vector[3]
    var roll = vector[4]
    var yaw = vector[5]
    var focallength = vector[6]

    let x0 = sin(pitch)
    let x1 = sin(yaw)
    let x2 = cos(pitch)
    let x4 = cos(yaw)
    let x9 = cos(roll)
    let x12 = sin(roll)

    var error = 0

    for(var i = 0; i < screen_points.length; ++i) {
      var screen_point = screen_points[i]
      var world_point = world_points[i]
      var screen_x = screen_point[0]
      var screen_y = screen_point[1]
      var world_x= world_point[0]
      var world_y = world_point[1]
      var world_z = world_point[2]
      let x3  =  world_x*x2
      
      let x5 = world_y*x2
      let x6 = x*x2
      let x7 = x2*y
      let x8 = -world_z*x0 - x0*z + x1*x3 + x1*x6 + x4*x5 + x4*x7
      
      let x10 = world_z*x2
      let x11 = x2*z
      
      let x13 = x1*x12
      let x14 = x4*x9
      let x15 = x0*x14
      let x16 = x13 + x15
      let x17 = x12*x4
      let x18 = x1*x9
      let x19 = x0*x18
      let x20 = -x17 + x19
      let x21 = world_x*x20 + world_y*x16 + x*x20 + x10*x9 + x11*x9 + x16*y
      let x22 = 1/x21
      let x23 = -focallength*x22*x8 - screen_y + width/2
      let x24 = x0*x13
      let x25 = x14 + x24
      let x26 = x0*x17 - x18
      let x27 = world_x*x25 + world_y*x26 + x*x25 + x10*x12 + x11*x12 + x26*y
      let x28 = -focallength*x22*x27 + height/2 - screen_x
      let x29 = 2*focallength*x2*x22
      let x30 = x17 - x19
      let x31 = x21**(-2)
      let x32 = 2*focallength*x31*x8
      let x33 = 2*focallength*x22
      let x34 = 2*focallength*x27*x31
      let x35 = -x13 - x15
      let x36 = x2*x9
      let x37 = -world_x*x16 - world_y*x30 - x*x16 - x30*y
      let x38 = -x14 - x24

      
      


      fprime[0] += (x23*(-x1*x29 - x30*x32) + x28*(-x25*x33 - x30*x34) ) / error_scale
      fprime[1] += (x23*(-x29*x4 - x32*x35) + x28*(-x26*x33 - x34*x35)) / error_scale
      fprime[2] += (x23*(x0*x33 + x32*x36) + x28*(-x12*x29 + x34*x36)) / error_scale
      fprime[3] += (x23*(-x32*x37 - x33*(-x1*x5 - x1*x7 + x3*x4 + x4*x6)) + x28*(-x33*(world_x*x26 + world_y*x38 + x*x26 + x38*y) - x34*x37)) / error_scale
      error += (x23**2 + x28**2) / error_scale
    }
    
    return error
  }
  
  return error_restricted
}



function solve_minimum(vector, screen_points, world_points) {
  var real_solution = {fx:9999999};
  window.cnt = 0;
  [0].forEach((yaw) => {
    vector = vector.slice()
    var loss = make_error_params(vector, screen_points, world_points)
    var solution = fmin.conjugateGradient(loss, [0, 0, -2.5, 0], {maxIterations: 300}) // vector[0], vector[1], vector[2], vector[5]])
    if(real_solution.fx > solution.fx){
      real_solution = solution
    }
  });
  console.log("iterations: ", window.cnt)
  console.log(real_solution);
  [[0, 0], [1, 1], [2, 2], [3, 5]].forEach((j)=>{
    vector[j[1]] = real_solution.x[j[0]]
  })
  return {v: vector, error: real_solution.fx}

}
function sortlines(lines, reference) {
  function value(line){
    return intersection(line, reference[0]).data32F[0]
  }
  return lines.sort((a, b) => value(a) - value(b))
}

function mod1(x) {
  return (x % 1 + 1) % 1
}
function distance(p1, p2){
  return Math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
}
function embed(l){
  return [Math.sin(2 * l[1]), Math.cos(2 * l[1])]
}  
function split(lines) {
  //window.trackingctx = document.getElementById("tracking").getContext("2d")


  clustermaker.k(2)
  clustermaker.iterations(209)
  var data = []
  lines.forEach(l => {
    data.push(embed(l))
    //the following is horrible, but avoids singularities in the GMM code I'm using
    /*gmm.addPoint(embed(l))
    gmm.addPoint([embed(l)[0], embed(l)[1] + .03])
    gmm.addPoint([embed(l)[0] + .03, embed(l)[1]])*/

  })
  clustermaker.data(data)
  var res = clustermaker.clusters()
  //gmm.runEM(7)
  var lines1 = []
  var lines2 = []
  //window.trackingctx.fillStyle = "rgba(255,255,255,1)"
  //        window.trackingctx.fillRect(0, 0, 128, 128)
  lines.forEach(l => {
    //if(gmm.predictNormalize(embed(l))[0] > .5){
    if(distance(embed(l), res[0].centroid) < distance(embed(l), res[1].centroid)) {
      lines1.push(l)

      //window.trackingctx.fillStyle = "rgba(255,0,0,1)"
      //window.trackingctx.fillRect(64  + 32 * embed(l)[0], 64 + 32 * embed(l)[1], 2, 2)

    } else {
      lines2.push(l)
      //window.trackingctx.fillStyle = "rgba(0,255,0,1)"
      //window.trackingctx.fillRect(64  + 32 * embed(l)[0], 64 + 32 * embed(l)[1], 2, 2)

    }
    
  })
  return [lines1, lines2]

}

var imu_yaw2grid_yaw_delta
var has_run_once = false
window.prune_strat = "mostvotes"
function getlines (array, imu_idx) {
  window.mat = cv.matFromArray(128, 128, cv.CV_32F, array)
  var gr = new cv.Mat()
  mat.convertTo(mat, cv.CV_8UC1)
  cv.threshold(mat, gr, .5, 256, cv.THRESH_BINARY_INV)
  window.gr = gr
  var skele = cv.Mat.zeros(128, 128, cv.CV_8UC1)
  var temp = cv.Mat.zeros(128, 128, cv.CV_8UC1)
  var elem = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(3, 3))
  var i
  for (i = 0; i < 7; i++) {
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
  drawLines(lines, dst, [76, 0, 0, 255])
  if (lines.rows < 2){
    cv.imshow('lines', dst);
    dst.delete()
    return
  }
  
  var lineDistMat = []
  for (let i = 0; i < lines.rows; ++i){
    lineDistMat.push([])
    for (let j = 0; j < lines.rows; ++j) {
      var rho1 = lines.data32F[i * 3];
      var theta1 = lines.data32F[i * 3 + 1];

      let rho2 = lines.data32F[j * 3];
      let theta2 = lines.data32F[j * 3 + 1];

      let distance1 = Math.sqrt((rho1 - rho2) * (rho1 - rho2) / (100 * 100) + (theta1 - theta2) * (theta1 - theta2))
      rho1 = rho1 * -1
      theta1 = theta1 - Math.PI
      let distance2 = Math.sqrt((rho1 - rho2) * (rho1 - rho2) / (100 * 100) + (theta1 - theta2) * (theta1 - theta2))
      //console.log(distance1, distance2, rho1, theta1)
      lineDistMat[i].push(Math.max(-distance1, -distance2))
    }
  }
  //console.table(lineDistMat)
  var preference = -.6;
  var cluster_result = apclust.getClusters(lineDistMat, {preference:preference, damping:.5})
  //console.table(lineDistMat)
  
  if (cluster_result.exemplars.length > 1){
    var lines_pruned = []
    if (window.prune_strat == "mostvotes"){
      var cluster_exemplar_lookup = []
      
      for (let j = 0; j < cluster_result.exemplars.length; ++j) {
        cluster_exemplar_lookup[cluster_result.exemplars[j]] = j
        lines_pruned.push([0, 0, 0])
      }
      
      for (let i = 0; i < lines.rows; ++i){
        var cluster_idx = cluster_exemplar_lookup[cluster_result.clusters[i]]
        
        if(lines.data32F[i * 3 + 2] > lines_pruned[cluster_idx][2]){
           let rho = lines.data32F[i * 3];
           let theta = lines.data32F[i * 3 + 1];
           let score = lines.data32F[i * 3 + 2];
           
           lines_pruned[cluster_idx] = [rho, theta, score]
        }

      }
      
      for (let j = 0; j < cluster_result.exemplars.length; ++j) {
        
        let rho = lines_pruned[j][0];
        let theta = lines_pruned[j][1];
        let a = Math.cos(theta);
        let b = Math.sin(theta);
        let x0 = a * rho;
        let y0 = b * rho;
        let startPoint = {x: x0 - 1000 * b, y: y0 + 1000 * a};
        let endPoint = {x: x0 + 1000 * b, y: y0 - 1000 * a};
        cv.line(dst, startPoint, endPoint, [255, 255, 0, 255]);
      }

    } else {


      for (let j = 0; j < cluster_result.exemplars.length; ++j) {
        let i = cluster_result.exemplars[j]
        let rho = lines.data32F[i * 3];
        let theta = lines.data32F[i * 3 + 1];
        lines_pruned.push([rho, theta])
        let a = Math.cos(theta);
        let b = Math.sin(theta);
        let x0 = a * rho;
        let y0 = b * rho;
        let startPoint = {x: x0 - 1000 * b, y: y0 + 1000 * a};
        let endPoint = {x: x0 + 1000 * b, y: y0 - 1000 * a};
        cv.line(dst, startPoint, endPoint, [255, 255, 0, 255]);
      }
    }

    /*let i1 = mat32FToArray(intersection(lines_pruned[0], lines_pruned[1]))
    let i2 = mat32FToArray(intersection(lines_pruned[1], lines_pruned[2]))
    console.log(i1)
    cv.line(dst, {x: i1[0][0], y:i1[1][0]}, {x: i2[0][0], y:i2[1][0]}, [0, 0, 255, 255])*/


    


    var split_lines = split(lines_pruned)


    drawLinesJ(split_lines[0], dst, [0, 255, 0, 255])
    drawLinesJ(split_lines[1], dst, [255, 0, 255, 255])

    if (lines_pruned.length < 3){
      cv.imshow('lines', dst);
      dst.delete()
      return
    }

    split_lines[0] = sortlines(split_lines[0], split_lines[1])
    split_lines[1] = sortlines(split_lines[1], split_lines[0])

    console.log(split_lines)

    var world_points = []
    var screen_points = []

    for(var i = 0; i < split_lines[0].length; ++i) {
      var l1 = split_lines[0][i]
      for( var j = 0; j < split_lines[1].length; ++j) {
        var l2 = split_lines[1][j]
        var x = intersection(l1, l2)
        x = [x.data32F[0], x.data32F[1] / window.webcam.webcamElement.videoWidth * window.webcam.webcamElement.videoHeight]
        screen_points.push(x)
        world_points.push([i - 1, j - 1, 0])

      } 
    }
    
    cv.imshow('lines', dst);
    var imu = events[imu_idx] || {
        alpha: 0,
        beta: 0,
        gamma: 0
      }
    var vector = [0, 0, -2.5, -imu.beta / 180 * Math.PI, -imu.gamma/ 180 * Math.PI, -imu.alpha/ 180 * Math.PI, (4/3) * 117]
    var res = solve_minimum(vector, screen_points, world_points)

    world_points = []
    screen_points = []

    for(var i = 0; i < split_lines[1].length; ++i) {
      var l1 = split_lines[1][i]
      for( var j = 0; j < split_lines[0].length; ++j) {
        var l2 = split_lines[0][j]
        var x = intersection(l1, l2)
        x = [x.data32F[0], x.data32F[1] / window.webcam.webcamElement.videoWidth * window.webcam.webcamElement.videoHeight]
        screen_points.push(x)
        world_points.push([i - 1, j - 1, 0])

      } 
    }

    var res2 = solve_minimum(vector, screen_points, world_points)
    var vector
    var error
    if(res2.error < res.error){
      vector = res2.v
      error = res2.error
    } else {
      vector = res.v
      error = res.error
    }

    
      
    var offset
    if (!has_run_once) {
      if (split_lines[0].length == 1 || split_lines[1].length == 1 ){
        return 
      }
      imu_yaw2grid_yaw_delta = vector[5] + imu.alpha / 180 * Math.PI
      console.log(imu_yaw2grid_yaw_delta)
      has_run_once = true
    } else {
      var correct_yaw_grid_coords = -imu.alpha / 180 * Math.PI + imu_yaw2grid_yaw_delta
      offset = correct_yaw_grid_coords - vector[5]

      var cos = Math.cos(offset)
      var sin = Math.sin(offset)
      var x = mod1( cos * vector[0] + sin * vector[1])
      var y = mod1(-sin * vector[0] + cos * vector[1])
      vector[0] = x
      vector[1] = y
      vector[5] = correct_yaw_grid_coords
    



    }
    
    if (error < .04 * screen_points.length){


      $("#transform").text(vector)
      window.transforms.push({'imu_idx': imu_idx, 'transform': vector, 'lines': split_lines})
    
      $("#error").text(error)
      $("#offset").text((offset * 180 / Math.PI) % 90)



      
      window.trackingctx = document.getElementById("tracking").getContext("2d")

      window.trackingctx.fillStyle = "rgba(255,255,255,1)"
      window.trackingctx.fillRect(0, 0, 128, 128)
      window.trackingctx.fillStyle = "rgba(255,0,0,1)"
      window.trackingctx.fillRect(128 * x, 128 - 128 * y, 4, 4)


      
    }

      
 
    
    dst.delete();
  }

}
