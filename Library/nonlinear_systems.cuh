#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <cassert>


/**
 * ���������� ������� ��� ���������� �������.
 * TODO: ���� � ������� ����������� ������� ������ �������� � �������� ������� (�������)
 * NOTE: ������ ������ ������ и�����
 * 
 * \param x - ������ �� ����������� x, y, z, ...
 * \param values - ������ �� ��������� a, b, c, ...
 * \param h - ��� ��������������
 * \return - ��������� ������������ � �������� ������ x
 */
template <class T>
__device__ void calculateNonlinearSystem(T* x, T* values, float h);



/**
 * ��������� �������� ���������� ������� � �����.
 * ����� �������� ��������� ���������� � ������ data
 * 
 * \param amountOfItearations - ���������� �������� ����������
 * \param x - ������ �� ����������� x, y, z, ...
 * \param values - ������ �� ��������� a, b, c, ...
 * \param h - ��� ��������������
 * \param valueLimit - ��������� ��������, ��� ���������� �������� ������� ��������� ������ � ���������� true
 * \param numberOfX - ����������, �� ������� ��������������� valueLimit � ���� ������ � data
 * \param prescaller - �������� �����������. ������ ���������� � �������� ������� prescaller ����� �������� � data
 * (�� ��������� prescaller = 1, ��� �������� ���������� �����������)
 * \param startIndexOfData - ��������� ������ ������� � ������ data (���� -1 - ������ �� ������������)
 * \param data - ������ data
 * \return ���� valueLimit ��� �������� - ������������ true. ����� false
 */
template <class T>
__device__ bool calculateLoopNonlinearSystem(int amountOfItearations, T* x, T* values, float h, int valueLimit,
	int numberOfX, int prescaller=1, int startIndexOfData = -1, T* data = nullptr);



/**
 * ������� ���� ����� �������� �����.
 * ����� ��������� ���-�� ��������� ����� ��� ������� �� ����� ������.
 * ���������� ����� (������ * ���) ������� �� ��������� �����
 * 
 * \param inputData - ������� ������ (��������� ������ ������ � ������� ������� �������)
 * \param startDataIndex - ��������� ������, � �������� ��������� ����� ����� � inputData
 * \param amountOfDataInBlock - ���-�� ������ � ����� ����� inputData (startDataIndex + amountOfDataInBlock = finishIndex)
 * \param outputDataPeakIndices - ������, ���� ���� ������ ��������� �������� �����
 * ��������! ������ ��������� � outputDataPeakIndices ����� ���������� ��������� �����!!!
 * \return ���������� ���������� ��������� �����
 * 
 */
template <class T>
__device__ int peakFinder(T* inputData, int startDataIndex, int amountOfDataInBlock, int* outputDataPeakIndices);



/**
 * ��������� ���������� �� amountOfElements ����� �� a �� b ������ data
 * 
 * \param intervals - ������ � ������� ������� ������� (2 ��������)
 * \param amountOfElements - ���-�� ���������
 * \param data - �������������� ������
 * \param startIndex - ��������� ������ ������ � data
 */
template <class T>
__host__ void linspace(T* intervals, int amountOfElements, T* data, int startIndex=0);



/**
 * ��������� ������ linspace. 
 * 
 * \param intervals - ������ � ������� ������� �������� (4 ��������)
 * \param amountOfElements - ���-�� ���������
 * \param data1 - ������ �������������� ������
 * \param data2 - ������ �������������� ������
 */
template <class T>
__host__ void linspace2D(
	T* intervals,
	int amountOfElements,
	T* data1, T* data2);



/**
 * ���������� ������ linspace. 
 * 
 * \param intervals - ������ � ������� ������� �������� (6 ���������)
 * \param amountOfElements - ���-�� ���������
 * \param data1 - ������ �������������� ������
 * \param data2 - ������ �������������� ������
 * \param data3 - ������ �������������� ������
 */
template <class T>
__host__ void linspace3D(
	T* intervals,
	int amountOfElements,
	T* data1, T* data2, T* data3);



/**
 * ���������� ������ ���������� �������� � ����
 * 
 * \param data - ������ � �������
 * \param peaksIndices - ������ � ��������� �����
 * \param paramValues - ���������� ��������
 * \param amountOfBlocks - ���������� ������ � ������� � ������� data
 * \param blockSize - ������������� ������ ������ ����� � data
 * \param outFileStream - �������� �����
 * \return  - ���� ������ ��������� ������. false - ���� ���.
 */
template <class T>
__host__ bool writingToFile1D(T* data, int* peaksIndices, T* paramValues, int amountOfBlocks, int blockSize, std::ofstream outFileStream);



/**
 * ������������� ���������� ���������� ������� �� �������� ����������.
 * ������������ �� 3� ���������� ����������
 * 
 * \param nPts - ���������� �������� 
 * \param tMax - ����� ������������� 
 * \param h - ��� ��������������
 * \param initialConditions - ������ � ���������� ���������
 * \param amountOfInitialContions - ���-�� ��������� �������
 * \param numberOfX - ����� ����������� ����������� (�������� 0 = x, 1 = y, etc.)
 * \param prePeakFinder - ���������� ������������ ����� ����������� ��������
 * \param data - �������������� ������
 * \param utilityData - ��������������� ������
 * \param valueLimitInCalculateNonlinearSystem - ����������� �������� �������, ����� �������� ������ �� ������������
 * \param modes - ������ � �������� ������ (��� 1D - ���� ��������, ��� 2D - ��� ��������, ��� 3D - 3 ��������)
 * \param params - ������ � �����������
 * \param amountOfParams - ���-�� ����������
 * \param prescaller - �������� �����������
 * \param paramValues1 - ��� �������� ��������� ����������� ���������
 * \param paramValues2 - ��� �������� ������� ��������� ����������� ��������� (��� 2D � 3D)
 * \param paramValues3 - ��� �������� �������� ��������� ����������� ��������� (��� 3D)
 */
template <class T>
__global__ void GPUComputationNLSystem(
	int nPts,
	int tMax,
	float h,
	T* initialConditions,
	int amountOfInitialContions,
	int numberOfX,
	int prePeakFinder,
	T* data,
	int* utilityData,
	int valueLimitInCalculateNonlinearSystem,
	int* modes,
	T* params,
	int amountOfParams,
	int prescaller,
	T* paramValues1,
	T* paramValues2 = nullptr,
	T* paramValues3 = nullptr
);



template <class T>
__host__ void bifurcation1D(
	int					nPts,
	int					tMax,
	float				h,
	T*					initialConditions,
	int					amountOfInitialContions,
	T*					paramRanges,
	int					numberOfX,
	int					prePeakFinder,
	int					valueLimitInCalculateNonlinearSystem,
	int*				modes,
	T*					params,
	int					amountOfParams,
	int					prescaller,
	std::string			outPath);



template <class T>
__host__ void slice(T* in, int a, int b, T* out);