#include "nonlinear_systems.cuh"
#include <cassert>

template <class T>
__device__ void calculateNonlinearSystem(T* x, T* values, float h)
{
	float localH1 = h * values[0];
	float localH2 = h * (1 - values[0]);

	x[0] = x[0] + localH1 * (-x[1] - x[2]);
	x[1] = (x[1] + localH1 * (x[0])) / (1 - values[1] * localH1);
	x[2] = (x[2] + localH1 * values[2]) / (1 - localH1 * (x[0] - values[3]));

	x[2] = x[2] + localH2 * (values[2] + x[2] * (x[0] - values[3]));
	x[1] = x[1] + localH2 * (x[0] + values[1] * x[1]);
	x[0] = x[0] + localH2 * (-x[1] - x[2]);
	return;
}



template <class T>
__device__ bool calculateLoopNonlinearSystem(int amountOfItearations, T* x, T* values, float h, int valueLimit,
	int numberOfX, int prescaller, int startIndexOfData, T* data)
{
	for (int i = 0; i < amountOfItearations; ++i)
	{
		// Запись в массив
		if (startIndexOfData != -1)
			data[startIndexOfData + i] = x[numberOfX];

		// Просеивание
		for (int j = 0; j < prescaller; ++j)
			calculateNonlinearSystem(x, values, h);

		if (x[numberOfX] > valueLimit)
		{
			return true;
		}
	}
	return false;
}



template <class T>
__device__ int peakFinder(T* inputData, int startDataIndex, int amountOfDataInBlock, int* outputDataPeakIndices)
{
	int amountOfFoundPeaks = 0;
	for (int i = startDataIndex + 1; i < startDataIndex + amountOfDataInBlock - 1; ++i)
	{
		// Если встретили чёткий пик - записываем его
		if (inputData[i] > inputData[i - 1] && inputData[i] < inputData[i + 1])
		{
			outputDataPeakIndices[startDataIndex + amountOfFoundPeaks + 1] = i;
			++amountOfFoundPeaks;
		}
		// Поиск предполагаемого пика (например случай: 2 4 6 6 3 1: нужно зафиксировать 3ий индекс как пик)
		else if (inputData[i] > inputData[i - 1] && inputData[i] == inputData[i + 1])
		{
			for (int j = i; j < startDataIndex + amountOfDataInBlock - 1; ++j)
			{
				// Если все-таки это не пик
				if (inputData[k] < inputData[k + 1])
				{
					break;
					i = k;
				}
				// Продолжаем искать пик
				if (inputData[k] == inputData[k + 1])
				{
					continue;
				}
				// Пик найден
				if (inputData[k] < inputData[k + 1])
				{
					outputDataPeakIndices[startDataIndex + amountOfFoundPeaks + 1] = k;
					++amountOfFoundPeaks;
					i = k + 1;
					break;
				}
			}
		}
	}
	outputDataPeakIndices[startDataIndex] = amountOfFoundPeaks;
	return amountOfFoundPeaks;
}



template <class T>
__host__ void linspace(T* intervals, int amountOfElements, T* data, int startIndex)
{
	assert(amountOfElements > 0);
	assert(intervals[1] >= intervals[0]);
	assert(data != nullptr);

	if (amountOfElements == 1)
	{
		data[0] = intervals[0];
		return;
	}

	T step = (intervals[1] - intervals[0]) / (amountOfElements - 1);
	for (int i = 0; i < amountOfElements; ++i)
		data[startIndex + i] = intervals[0] + i * step;
	return;
}



template <class T>
__host__ void linspace2D(
	T* intervals,
	int amountOfElements,
	T* data1, T* data2)
{
	assert(data1 != nullptr);
	assert(data2 != nullptr);

	T* tempData = new T[amountOfElements];
	linspace<T>(intervals + 2, amountOfElements, tempData);

	for (int i = 0; i < amountOfElements; ++i)
	{
		linspace<T>(intervals, amountOfElements, data1, i * amountOfElements);
		for (int j = 0; j < amountOfElements; ++j)
			data2[amountOfElements * i + j] = tempData[i];
	}

	delete[] tempData;
}



template <class T>
__host__ void linspace3D(
	T* intervals,
	int amountOfElements,
	T* data1, T* data2, T* data3)
{
	assert(data1 != nullptr);
	assert(data2 != nullptr);
	assert(data3 != nullptr);

	T* tempData2 = new T[amountOfElements];
	T* tempData3 = new T[amountOfElements];

	linspace<T>(intervals + 2, amountOfElements, tempData2);
	linspace<T>(intervals + 4, amountOfElements, tempData3);

	for (int k = 0; k < amountOfElements; ++k)
		for (int i = 0; i < amountOfElements; ++i)
		{
			linspace(intervals, amountOfElements,
				data1, i * amountOfElements + k * amountOfElements * amountOfElements);

			for (int j = 0; j < amountOfElements; ++j)
			{
				data2[amountOfElements * amountOfElements * k + amountOfElements * i + j] = tempData2[i];
				data3[amountOfElements * amountOfElements * k + amountOfElements * i + j] = tempData3[k];
			}
		}

	delete[] tempData2;
	delete[] tempData3;
}



template <class T>
__host__ bool writingToFile1D(T* data, int* peaksIndices, T* paramValues, int amountOfBlocks, int blockSize, std::ofstream outFileStream)
{
	for (int i = 0; i < amountOfBlocks; ++i)
		for (size_t j = 1; j < peaksIndices[i * blockSize]; ++j)
			if (outFileStream.is_open())
				outFileStream << paramValues[i] << ", " << data[peaksIndices[j]] << '\n';
			else
				return false;
	return true;
}



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
	T* paramValues2,
	T* paramValues3
)
{
	// Индекс потока
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= nPts)
		return;

	// Расчет кол-ва полезных и отбрасываемых точек
	int amountOfSkipPoints = prePeakFinder / h;
	int amountOfTPoints = tMax / h / prescaller;

	// Копирование начальных условий
	T* x = new T[amountOfInitialContions];
	for (int i = 0; i < amountOfInitialContions; ++i)
		x[i] = initialConditions[i];

	// Копирование начальных параметров
	float* localParam = new float[amountOfParams];
	for (int i = 0; i < amountOfParams; ++i)
		localParam[i] = params[i];

	// Установка изменяемых параметров по режиму работы
	localParam[modes[0]] = paramValues1[idx];

	if (paramValues2 != nullptr)
		localParam[modes[1]] = paramValues2[idx];

	if (paramValues3 != nullptr)
		localParam[modes[2]] = paramValues3[idx];

	// Убираем переходные процессы - отбрасываем начальыне точки
	if (calculateLoopNonlinearSystem(amountOfSkipPoints, x, localParam, h, valueLimitInCalculateNonlinearSystem,
		numberOfX, prescaller))
	{
		// Если был превышен valueLimitInCalculateNonlinearSystem
		delete[] localParam;
		utilityData[idx * amountOfTPoints] = -1;
		return;
	}

	// Вычисляем системы по заданным параметрам
	if (calculateLoopNonlinearSystem(amountOfTPoints, x, localParam, h, valueLimitInCalculateNonlinearSystem,
		numberOfX, prescaller, idx * amountOfTPoints, data))
	{
		// Если был превышен valueLimitInCalculateNonlinearSystem
		delete[] localParam;
		utilityData[idx * amountOfTPoints] = -1;
		return;
	}

	delete[] x;
	delete[] localParam;

	return;
}



template <class T>
__host__ void bifurcation1D(
	int					nPts,
	int					tMax,
	float				h,
	T* initialConditions,
	int					amountOfInitialContions,
	T* paramRanges,
	int					numberOfX,
	int					prePeakFinder,
	int					valueLimitInCalculateNonlinearSystem,
	int* modes,
	T* params,
	int					amountOfParams,
	int					prescaller,
	std::string			outPath)
{
	int amountOfTPoints = tMax / h / prescaller;

	float* globalParamValues = (T*)malloc(sizeof(T) * nPts);
	linspace<T>(paramRanges, nPts, globalParamValues);

	size_t freeMemory;
	size_t totalMemory;

	cudaMemGetInfo(&freeMemory, &totalMemory);

	freeMemory *= 0.7;

	//! Для одномерной бифуркационной диаграммы нам понадобиться:
	//! data			- nPtsLimiter * amountOfTPoints * sizeof(T)
	//! utilityData		- nPtsLimiter * amountOfTPoints * sizeof(int)
	//! Остальным можно пренебречь
	//! nPtsLimiter * amountOfTPoints * (sizeof(T) + sizeof(int)) = Память которая нужна = freeMemory
	//! nPtsLimiter = freeMemory / amountOfTPoints * (sizeof(T) + sizeof(int));

	size_t nPtsLimiter = freeMemory / (amountOfTPoints * (sizeof(T) + sizeof(int)));
	assert(nPtsLimiter > 0);

	T* h_data;
	int* h_utilityData;
	T* h_paramValues;

	T* d_data;
	int* d_utilityData;
	T* d_paramValues;
	T* d_params;
	T* d_initialConditions;

	cudaMalloc((void**)&d_params, amountOfParams * sizeof(T));
	cudaMalloc((void**)&d_initialConditions, amountOfInitialContions * sizeof(T));
	cudaMemcpy(d_params, params, amountOfParams * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(d_initialConditions, initialConditions, amountOfInitialContions * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);

	size_t amountOfIteration = (size_t)std::ceilf((float)nPts / (float)nPtsLimiter);

	std::ofstream outFileStream;
	outFileStream.open(outPath);

	for (size_t i = 0; i < amountOfIteration; ++i)
	{
		if (i == amountOfIteration - 1)
		{
			h_paramValues = (T*)malloc((nPts - nPtsLimiter * i) * sizeof(T));

			slice(globalParamValues, nPtsLimiter * i, nPts, h_paramValues);
			nPtsLimiter = nPts - (nPtsLimiter * i);
		}
		else
		{
			h_paramValues = (T*)malloc(((nPtsLimiter * i + nPtsLimiter) - nPtsLimiter * i) * sizeof(T));
			slice(globalParamValues, nPtsLimiter * i, nPtsLimiter * i + nPtsLimiter, h_paramValues);
		}


		h_data = (T*)malloc(nPtsLimiter * amountOfTPoints * sizeof(T));
		h_utilityData = (int*)malloc(nPtsLimiter * amountOfTPoints * sizeof(int));

		cudaMalloc((void**)&d_data, nPtsLimiter * amountOfTPoints * sizeof(T));
		cudaMalloc((void**)&d_utilityData, nPtsLimiter * amountOfTPoints * sizeof(int));
		cudaMalloc((void**)&d_paramValues, nPtsLimiter * sizeof(T));

		cudaMemcpy(d_paramValues, h_paramValues, nPtsLimiter * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);

		int blockSize;
		int minGridSize;
		int gridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, GPUComputationNLSystem<T>, 0, nPtsLimiter);
		gridSize = (nPtsLimiter + blockSize - 1) / blockSize;



		//Call CUDA func
		GPUComputationNLSystem<T> < < <gridSize, blockSize > > > (
			nPts,
			tMax,
			h,
			initialConditions,
			amountOfInitialContions,
			numberOfX,
			prePeakFinder,
			d_data,
			d_utilityData,
			valueLimitInCalculateNonlinearSystem,
			modes,
			params,
			amountOfParams,
			prescaller,
			d_paramValues
			);



		cudaMemcpy(h_data, d_data, amountOfTPoints * nPtsLimiter * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaMemcpy(h_utilityData, d_utilityData, amountOfTPoints * nPtsLimiter * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		cudaFree(d_data);
		cudaFree(d_utilityData);
		cudaFree(d_paramValues);

		writingToFile1D(h_data, h_utilityData, h_paramValues, nPtsLimiter, amountOfTPoints, outFileStream);

		std::free(h_data);
		std::free(d_utilityData);
		std::free(d_paramValues);
	}
	cudaFree(d_params);
	cudaFree(d_initialConditions);
	std::free(globalParamValues);

	outFileStream.close();
}

template <class T>
__host__ void slice(T* in, int a, int b, T* out)
{
	assert(b - a < 0);
	for (size_t i = 0; i < b - a; ++i)
		out[i] = in[a + i];
}