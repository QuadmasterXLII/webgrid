import * as utils from "./utils"

var P
var x 
var t_last = Date.now() / 1000

let q = .25

let position_stddev = .04 // meters
let acc_stddev = .1
let meters_per_foot = .3048

var position_initialized = false

var callback = () => {}

export function setCallback(f) {
    callback = f
}


export function init() {
     P = [cv.Mat.eye(3, 3, cv.CV_32F), cv.Mat.eye(3, 3, cv.CV_32F), cv.Mat.eye(3, 3, cv.CV_32F)]
     x = [cv.Mat.zeros(3, 1, cv.CV_32F), cv.Mat.zeros(3, 1, cv.CV_32F), cv.Mat.zeros(3, 1, cv.CV_32F)]
     position_initialized = false
}


export function update_position(vector, time) {
    position_initialized = true
    var dt = time - t_last
    console.log("pos", dt)
    t_last = time



    let A = cv.matFromArray(3, 3, cv.CV_32F, [1, dt, .5 * dt * dt,
                                          0, 1,  dt,
                                          0, 0, 1])
    let H = cv.matFromArray(1, 3, cv.CV_32F, [1, 0, 0])

    let Q =  cv.matFromArray(3, 3, cv.CV_32F, [dt * dt * dt * dt / 4, dt * dt * dt / 2, dt * dt / 2,
                                              dt * dt * dt / 2,      dt * dt,          dt,
                                              dt * dt / 2,           dt,               1].map(x=>x * q))
    
    let R = cv.matFromArray(1, 1, cv.CV_32F, [position_stddev * position_stddev])
    for(var j = 0; j < 3; ++j){



        var x_pred = utils.matmul(A, x[j])

        var offset = x_pred.data32F[0] - vector[j] * 2 * meters_per_foot

        while ( Math.abs(offset) > .5 * 2 * meters_per_foot && j != 2) {
            vector[j] += Math.sign(offset)
            offset = x_pred.data32F[0] - vector[j] * 2 * meters_per_foot
        }

        var p_pred = utils.add( utils.matmul(utils.matmul(A, P[j]), A.t()), Q)

        var y_tilde = utils.sub(cv.matFromArray(1, 1, cv.CV_32F, [vector[j] * 2 * meters_per_foot]), utils.matmul(H, x_pred))

        var S = utils.add(utils.matmul(utils.matmul(H, p_pred), H.t()), R)

        var K = utils.matmul(utils.matmul(p_pred, H.t()), S.inv(cv.DECOMP_LU))

        x[j] = utils.add(x_pred, utils.matmul(K, y_tilde))

        P[j] = utils.matmul(utils.sub(cv.Mat.eye(3, 3, cv.CV_32F), utils.matmul(K, H)), p_pred)
    }
    //callback(x)
}

export function update_imu(orientation, acceleration, imu_yaw_delta, time) {
    if(!x || !orientation || !imu_yaw_delta || !position_initialized) {
        return
    }
    var acc = world_acc(orientation, acceleration, imu_yaw_delta)


    var dt = time - t_last
    console.log("acc", dt)
    t_last = time

    let A = cv.matFromArray(3, 3, cv.CV_32F, [1, dt, .5 * dt * dt,
                                          0, 1,  dt,
                                          0, 0, 1])
    let H = cv.matFromArray(1, 3, cv.CV_32F, [0, 0, 1])

    let Q = cv.matFromArray(3, 3, cv.CV_32F, [dt * dt * dt * dt / 4, dt * dt * dt / 2, dt * dt / 2,
                                              dt * dt * dt / 2,      dt * dt,          dt,
                                              dt * dt / 2,           dt,               1].map(x=>x * q))

    
    let R = cv.matFromArray(1, 1, cv.CV_32F, [acc_stddev * acc_stddev])
    for(var j = 0; j < 3; ++j){



        var x_pred = utils.matmul(A, x[j])

        var p_pred = utils.add( utils.matmul(utils.matmul(A, P[j]), A.t()), Q)

        var y_tilde = utils.sub(cv.matFromArray(1, 1, cv.CV_32F, [acc.data32F[j]]), utils.matmul(H, x_pred))

        var S = utils.add(utils.matmul(utils.matmul(H, p_pred), H.t()), R)

        var K = utils.matmul(utils.matmul(p_pred, H.t()), S.inv(cv.DECOMP_LU))

        x[j] = utils.add(x_pred, utils.matmul(K, y_tilde))

        P[j] = utils.matmul(utils.sub(cv.Mat.eye(3, 3, cv.CV_32F), utils.matmul(K, H)), p_pred)
    }
    callback(x)
}

function world_acc(orientation, acceleration, imu_yaw_delta) {

    var sin = Math.sin
    var cos = Math.cos
    var a = -orientation.beta / 180 * Math.PI
    var b = -orientation.gamma / 180 * Math.PI
    var c = -orientation.alpha / 180 * Math.PI + imu_yaw_delta
    var rotmat = cv.matFromArray(3, 3, cv.CV_32F, [
      sin(a)*sin(b)*sin(c) + cos(b)*cos(c), sin(a)*sin(b)*cos(c) - sin(c)*cos(b), sin(b)*cos(a),
      sin(c)*cos(a),                        cos(a)*cos(c),       -sin(a),
      sin(a)*sin(c)*cos(b) - sin(b)*cos(c), sin(a)*cos(b)*cos(c) + sin(b)*sin(c), cos(a)*cos(b)
    ]).inv(cv.DECOMP_LU)

    return utils.matmul(rotmat, cv.matFromArray(3, 1, cv.CV_32F, [-acceleration.x, -acceleration.y, -acceleration.z]))
    //return cv.matFromArray(3, 1, cv.CV_32F, [-acceleration.x, -acceleration.y, -acceleration.z])
    

}