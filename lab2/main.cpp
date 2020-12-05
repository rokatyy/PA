//
// Created by Ekaterina Rogatova on 05.12.2020.
//

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <math.h>

#define size 2

using namespace std;

double **SerialMatrixMulti(double A[size][size], double B[size][size])
{
    int i,j,k;
    double **C = new double * [size];
    for (i = 0; i < size; i++) {
        C[i] = new double [size];
    }
   for (i=0; i<size; i++) {
        for (j=0; j<size; j++){
            C[i][j] = 0;
            for (k=0; k<size; k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
    return C;

}

int main(){
    double A[size][size] = {{1,1},{1,1}};
    double B[size][size] = {{1,1},{1,1}};
    double **C;

    C = SerialMatrixMulti(A,B);

    return 0;
}