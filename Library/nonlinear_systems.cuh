#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <cassert>


/**
 * Возвращает решение для нелинейной системы.
 * TODO: Надо в функции реализовать решение систем заданных в польской нотации (парсинг)
 * NOTE: Сейчас решает только Рёслера
 * 
 * \param x - Массив на неизвестные x, y, z, ...
 * \param values - Массив на параметры a, b, c, ...
 * \param h - Шаг интегрирования
 * \return - Результат записывается в исходный массив x
 */
template <class T>
__device__ void calculateNonlinearSystem(T* x, T* values, float h);



/**
 * Вычисляет значения нелинейной системы в цикле.
 * Может записать результат вычислений в массив data
 * 
 * \param amountOfItearations - Количество итераций вычислений
 * \param x - Массив на неизвестные x, y, z, ...
 * \param values - Массив на параметры a, b, c, ...
 * \param h - Шаг интегрирования
 * \param valueLimit - Пороговое значение, при превышении которого функция завершает работу и возвращает true
 * \param numberOfX - Переменная, по которой просматривается valueLimit и идет запись в data
 * \param prescaller - Параметр просеивания. Каждое вычисление с индексом кратным prescaller будет записано в data
 * (По умолчанию prescaller = 1, что означает отсутствие просеивания)
 * \param startIndexOfData - Стартовый индекс замписи в массив data (Если -1 - запись не производится)
 * \param data - Массив data
 * \return Если valueLimit был превышен - возвращается true. Иначе false
 */
template <class T>
__device__ bool calculateLoopNonlinearSystem(int amountOfItearations, T* x, T* values, float h, int valueLimit,
	int numberOfX, int prescaller=1, int startIndexOfData = -1, T* data = nullptr);



/**
 * Находит пики среди заданных даных.
 * Также возаращет кол-во найденных пиков для каждого из блока данных.
 * Возвращает время (индекс * шаг) каждого из найденных пиков
 * 
 * \param inputData - Входные данные (несколько подряд идущих в массиве решений системы)
 * \param startDataIndex - Начальный индекс, с которого начаинаем поиск пиков в inputData
 * \param amountOfDataInBlock - Кол-во данных в одном блоке inputData (startDataIndex + amountOfDataInBlock = finishIndex)
 * \param outputDataPeakIndices - Массив, куда идет запись найденных индексов пиков
 * ВНИМАНИЕ! ПЕРВЫМ ЭЛЕМЕНТОМ В outputDataPeakIndices БУДЕТ КОЛИЧЕСТВО НАЙДЕННЫХ ПИКОВ!!!
 * \return Возвращает количество найденных пиков
 * 
 */
template <class T>
__device__ int peakFinder(T* inputData, int startDataIndex, int amountOfDataInBlock, int* outputDataPeakIndices);



/**
 * Заполняет равномерно на amountOfElements чисел от a до b массив data
 * 
 * \param intervals - Нижняя и верхняя границы массива (2 элемента)
 * \param amountOfElements - Кол-во элементов
 * \param data - Результирующий массив
 * \param startIndex - Стартовый индекс записи в data
 */
template <class T>
__host__ void linspace(T* intervals, int amountOfElements, T* data, int startIndex=0);



/**
 * Двумерная версия linspace. 
 * 
 * \param intervals - Нижние и верхние границы массивов (4 элемента)
 * \param amountOfElements - Кол-во элементов
 * \param data1 - Первый результирующий массив
 * \param data2 - Второй результирующий массив
 */
template <class T>
__host__ void linspace2D(
	T* intervals,
	int amountOfElements,
	T* data1, T* data2);



/**
 * Трехмерная версия linspace. 
 * 
 * \param intervals - Нижние и верхние границы массивов (6 элементов)
 * \param amountOfElements - Кол-во элементов
 * \param data1 - Первый результирующий массив
 * \param data2 - Второй результирующий массив
 * \param data3 - Третий результирующий массив
 */
template <class T>
__host__ void linspace3D(
	T* intervals,
	int amountOfElements,
	T* data1, T* data2, T* data3);



/**
 * Записывает данные одномерной диграммы в файл
 * 
 * \param data - Массив с данными
 * \param peaksIndices - Массив с индексами пиков
 * \param paramValues - Переменный параметр
 * \param amountOfBlocks - Количество блоков в массиве с данными data
 * \param blockSize - Фиксированный размер одного блока в data
 * \param outFileStream - Файловый поток
 * \return  - Если запись произошла успшно. false - если нет.
 */
template <class T>
__host__ bool writingToFile1D(T* data, int* peaksIndices, T* paramValues, int amountOfBlocks, int blockSize, std::ofstream outFileStream);



/**
 * Многопоточное вычисление нелинейной системы по заданным параметрам.
 * Поддерживает до 3х изменяемых параметров
 * 
 * \param nPts - Разрешение диграммы 
 * \param tMax - Время моделирования 
 * \param h - Шаг интегрирования
 * \param initialConditions - Массив с начальными условиями
 * \param amountOfInitialContions - Кол-во начальных условий
 * \param numberOfX - Номер вычисляемой неизвестной (Например 0 = x, 1 = y, etc.)
 * \param prePeakFinder - Количество пропускаемых точек переходного процесса
 * \param data - Результирующий массив
 * \param utilityData - Вспомогательный массив
 * \param valueLimitInCalculateNonlinearSystem - Критическое значение сигнала, после которого расчет не производится
 * \param modes - Массив с режимами работы (При 1D - одно значение, при 2D - два значения, при 3D - 3 значения)
 * \param params - Массив с параметрами
 * \param amountOfParams - Кол-во параметров
 * \param prescaller - Параметр просеивания
 * \param paramValues1 - Все значения заданного изменяемого параметра
 * \param paramValues2 - Все значения второго заданного изменяемого параметра (Для 2D и 3D)
 * \param paramValues3 - Все значения третьего заданного изменяемого параметра (Для 3D)
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