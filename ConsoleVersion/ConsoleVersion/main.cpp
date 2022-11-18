#include "../../Library/nonlinear_systems.cuh"
//#include "../../Library/nonlinear_systems.cu"
#include <iostream>

int main()
{
	float* initialConditions = new float[3] {0.1, 0.1, 0.1};
	float* paramRanges = new float[2] {0.05, 0.35};
	int* modes = new int[1] {1};
	float* params = new float[4] {0.5, 0.2, 0.2, 5.7};

	bifurcation1D<float>(1000, 1000, 0.01f, initialConditions, 3, paramRanges, 0, 2000, 10000, modes, params, 4, 1, "mat.csv");
	return 0;
}
