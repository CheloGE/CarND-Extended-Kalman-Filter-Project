#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::max;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict()
{
    /**
   * predict the state
   */
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
    /**
   * update the state by using Kalman Filter equations
   */
    // calculate the measurement error vector
    y_ = z - H_ * x_;
    // calculate the covatiance matrix of the measurement
    S_ = H_ * P_ * H_.transpose() + R_;
    // calculate the kalman gain
    K_ = P_ * H_.transpose() * S_.inverse();
    // set Identity matrix
    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
    // Now lest proceed with the estimation correction section
    x_ += K_ * y_;
    P_ = (I - K_ * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
    /**
   * update the state by using Extended Kalman Filter equations
   */
    // calculate the measurement error vector
    // Here we use a mapping function since we have a non linear model
    y_ = z - h(x_);
    // Normalize angle to be between -2pi and 2pi
    while (y_(1) > M_PI)
        y_(1) -= 2 * M_PI;
    while (y_(1) < -M_PI)
        y_(1) += 2 * M_PI;
    // calculate the covatiance matrix of the measurement
    // Bear in mind that H is a jacobian
    S_ = H_ * P_ * H_.transpose() + R_;
    // calculate the kalman gain
    K_ = P_ * H_.transpose() * S_.inverse();
    // set Identity matrix
    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
    // Now lest proceed with the estimation correction section
    x_ += K_ * y_;
    P_ = (I - K_ * H_) * P_;
}

VectorXd KalmanFilter::h(const VectorXd &x_in)
{

    //// extract state variavles
    float px = x_in(0);
    float py = x_in(1);
    float vx = x_in(2);
    float vy = x_in(3);
    // map to polar coordinates
    float rho = sqrt(px * px + py * py);
    float phi = atan2(py, px);
    float rho_dot = (px * vx + py * vy) / max(rho, 0.0001f); // to avoid division by 0
    // return estimates in polar coordinates
    VectorXd polar(3);
    polar << rho, phi, rho_dot;
    return polar;
}