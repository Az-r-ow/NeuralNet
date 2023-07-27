#pragma once

#include <cmath>
#include <iostream>
#include <random>

/**
 * Rand double generator that uses the Mersenne Twister algo
 */
double mtRand(double min, double max);

/* Activation Functions */
double relu(double x);

double sigmoid(double x);

/* Mathematical functions */
/**
 * Returns the square of x
 */
double Sqr(double x);