#pragma once

#include <cmath>
#include <iostream>
#include <random>

/**
 * Rand double generator that uses the Mersenne Twister algo
 */
double mtRand(double min, double max);

double relu(double x);

double sigmoid(double x);