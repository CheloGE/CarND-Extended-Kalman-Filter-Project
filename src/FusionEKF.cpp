#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF()
{
    is_initialized_ = false;

    previous_timestamp_ = 0;

    /**
     * Initializing EKF parameters
     * 
     */

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix - laser
    R_laser_ << 0.0225, 0,
        0, 0.0225;

    //measurement covariance matrix - radar
    R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

    //Initialize H_laser matrix that maps state variables to measurements
    // since laser only measures the position we only map position values
    H_laser_ << 1, 0, 0, 0,
        0, 1, 0, 0;

    // Initialize the covariance matrix of the prediction process.
    // the diagonal contains the uncertainties. Therefore we will set a higher number for velocities and a lower ones for positions
    // The other values are set as zero since we assume there is no initial correlation between
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

    // initialize the state transition matrix F
    // we will set 1 in the positions of state values
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    ekf_.Q_ = MatrixXd(4, 4);

    noise_ax = 9;
    noise_ay = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
    /**
   * Initialization
   */
    if (!is_initialized_)
    {
        /**
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     * To do that we need to convert radar from polar to cartesian coordinates.
     */

        // first measurement
        // we initialize states with ones to avoid changing transformations.
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);
        ekf_.x_ << 1, 1, 1, 1;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
        {
            //
            // and initialize state.

            /** 
      *  Extract polar measurements from radar 
      */
            // ro is the magnitude of the polar vector from source to measured object
            float rho = measurement_pack.raw_measurements_[0];
            // phi is the angle of the polar vector from source to measured object
            float phi = measurement_pack.raw_measurements_[1];
            // ro_dot is the velocity of the polar vector projected from the measured object to the source
            float rho_dot = measurement_pack.raw_measurements_[2];

            // Convert radar from polar to cartesian coordinates
            float px = rho * cos(phi);
            float py = rho * sin(phi);
            // vx and vy cannot be mapped directly from single measurement as we don't know the direction of the object yet
            // Therefore, we will estimate it as it would directly away from us
            float vx = rho_dot * cos(phi);
            float vy = rho_dot * sin(phi);

            //we set the state directly from measurements without considering predictions
            //as it is the 1st state from our Kalman Filter.
            ekf_.x_ << px, py, vx, vy;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
        {
            /** 
      *  Extract  measurements from Lidar 
      */
            float px = measurement_pack.raw_measurements_[0];
            float py = measurement_pack.raw_measurements_[0];
            // since lidar only measures position we will update only px and py and set vx and vy as zero.
            ekf_.x_ << px, py, 0.0, 0.0;
        }

        // Update last measurement
        previous_timestamp_ = measurement_pack.timestamp_;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /**
   * Prediction
   */

    /**
     * Prediction section
   */

    // we first get the time difference between measurement
    // we need it in seconds hence we need to divide it by 10^6
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    // update previous time
    previous_timestamp_ = measurement_pack.timestamp_;

    // uodate the state transition matrix F according to the model equations for movement
    // px' = px + vx*dt
    // we assume that  velocity will remain constant so we will set v'=v
    // we  only set the dt as the F remaining values state constant and were initialized once.
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    // as we are assuming a constant velocity we should consider the error this assumtion could create
    // therefore we update our Q (model noise) matrix considering acceleration noise comming from ax and ay
    float dt4 = dt * dt * dt * dt; // dt to the 4th power
    float dt3 = dt * dt * dt;      // de to the 3rd power
    float dt2 = dt * dt;           // dt squarred
    float noise_ax_2 = noise_ax * noise_ax;
    float noise_ay_2 = noise_ay * noise_ay;
    ekf_.Q_ << dt4 * noise_ax_2 / 4, 0, dt3 * noise_ax_2 / 2, 0,
        0, dt4 * noise_ay_2 / 4, 0, dt3 * noise_ay_2 / 2,
        dt3 * noise_ax_2 / 2, 0, dt2 * noise_ax_2, 0,
        0, dt3 * noise_ay_2 / 2, 0, dt2 * noise_ay_2;

    ekf_.Predict();

    /**
   * Update AKA correction section
   */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
        /**
         *  radar update process
         */
        // first we need the R value which is taken from sensor's uncertainty
        // usually provided by manufacturer
        ekf_.R_ = R_radar_;
        // since this is a non-linear sensor model, we must linearize it with taylor series around a point.
        // To do this we need to calculate the Jacobian upto the 1st derivative
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        // finally we update the EKF based on the RADAR measurement AKA "z"
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    }
    else
    {
        /**
         *  Laser update process
         */
        // R value which is taken from sensor's uncertainty
        // usually provided by manufacturer
        ekf_.R_ = R_laser_;
        // This is a linear sensor model, so we can get a matrix based on the mapping of measurements to state variables
        ekf_.H_ = H_laser_;
        // finally we update the KF based on the RADAR measurement AKA "z"
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
