#include "tools.h"
#include <iostream>
#include <stdexcept>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
    /**
   * TODO: Calculate the RMSE here.
   */

    VectorXd residual(4);
    VectorXd accumulateRes(4);
    VectorXd mean(4);
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    accumulateRes << 0, 0, 0, 0;
    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() == 0)
    {
        throw std::invalid_argument("estimation vector size is empty");
    }
    else if (estimations.size() != ground_truth.size())
    {
        throw std::invalid_argument("estimation vector size differs from ground truth");
    }
    else
    {
        // Accumulate squared residuals
        for (unsigned int i = 0; i < estimations.size(); ++i)
        {
            residual = estimations[i] - ground_truth[i];
            residual = residual.array() * residual.array();
            accumulateRes = accumulateRes.array() + residual.array();
        }
        // calculate the mean
        mean = accumulateRes.array() / estimations.size();
        // calculate the squared root
        rmse = mean.array().sqrt();
    }

    // return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
    /**
   * TODO:
   * Calculate a Jacobian here.
   */
    MatrixXd Hj(3, 4);
    // recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    // check division by zero
    if (px == 0 && py == 0)
    {
        throw std::invalid_argument("singularity found division by zero");
        Hj << 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;
        return Hj;
    }
    // compute the Jacobian matrix
    else
    {
        Hj << px / sqrt(px * px + py * py), py / sqrt(px * px + py * py), 0, 0,
            -py / (px * px + py * py), px / (px * px + py * py), 0, 0,
            py * (vx * py - vy * px) / sqrt((px * px + py * py) * (px * px + py * py) * (px * px + py * py)),
            px * (vy * px - vx * py) / sqrt((px * px + py * py) * (px * px + py * py) * (px * px + py * py)),
            px / sqrt(px * px + py * py), py / sqrt(px * px + py * py);
    }

    return Hj;
}
