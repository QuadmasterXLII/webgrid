
export function matmul(A, B){
    let res = new cv.Mat()
    cv.gemm(A, B, 1, cv.Mat.zeros(A.rows, B.cols, cv.CV_32F), 0, res, 0)
    return res
}

export function add(A, B){
    let res = new cv.Mat()
    let mask = new cv.Mat()
    let dtype = -1
    cv.add(A, B, res, mask, dtype)
    return res
}

export function sub(A, B){
    let res = new cv.Mat()
    let mask = new cv.Mat()
    let dtype = -1
    cv.subtract(A, B, res, mask, dtype)
    return res
}


export function intersection(line1, line2){
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
    //let res = new cv.Mat()
    //cv.gemm(A.inv(cv.DECOMP_LU), b, 1, cv.matFromArray(2, 1, cv.CV_32F, [0, 0]), 0, res, 0)

    
    return matmul(A.inv(cv.DECOMP_LU), b)
}

export function drawLines(lines, mat, color) {
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
export function drawLinesJ(lines, mat, color) {
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
export function mat32FToArray(mat) {
  var res = []
  for (let i = 0; i < mat.rows; ++i) {
    res.push([])
    for (let j = 0; j < mat.cols; ++j) {
      res[i].push(mat.data32F[i * mat.cols + j])
    }
  }
  return res
}
var input_shape = 128
export function project_points(vector, world_points){
  var sin = Math.sin
  var cos = Math.cos
  var pi = Math.PI
  var height = input_shape
  var width = input_shape / window.webcam.webcamElement.videoWidth * window.webcam.webcamElement.videoHeight



  var x = vector[0]
  var y = vector[1]
  var z = vector[2]
  var pitch =vector[3]
  var roll = vector[4]
  var yaw = vector[5]
  var focallength = vector[6]

  var screen_points = []

  let x0  =  sin(roll)
  let x1  =  cos(pitch)

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
  


  for(var i = 0; i < world_points.length; ++i) {
    var world_point = world_points[i]
    var world_x= world_point[0]
    var world_y = world_point[1]
    var world_z = world_point[2]
    let x2  =  world_z*x1
    let x_19 = x*x15 + x14*y + x2*x4 + x3*x4
    let x16  =  focallength/(world_x*x15 + world_y*x14 + x_19)
    screen_points.push([width/2 - x16*(world_x*x10 + world_y*x13 + x*x10 + x0*x2 + x0*x3 + x13*y), 
      (height/2 - x16*(world_x*x17 + world_y*x18 - world_z*x7 + x*x17 + x18*y - x7*z)) * window.webcam.webcamElement.videoWidth / window.webcam.webcamElement.videoHeight ])

  }
  return screen_points
}

export function mod1(x) {
  return (x % 1 + 1) % 1
}
export function distance(p1, p2){
  return Math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
}
