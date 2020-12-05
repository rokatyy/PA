//#include "/usr/local/opt/libomp/include/omp.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <math.h>
using namespace std;


#define N 10000000
#define NUM_THREADS 2000

double f(double x) {
    return 1 / (x * x) * sin(1 / x) * sin(1 / x);

}
double IntegralClassic(double a, double b, int n){
    double step = (b-a)/n;
    double x, integral=0;
    for (int i=1;i<n-1;i++){
        x=a+i*step;
        integral += 2 * (f(x));
    }
    return integral * step / 2;
}

double IntegralCritical(double a, double b, int n) {
    double step = (b - a) / n;
    double x, integral = 0;
    int i;
    #pragma omp parallel for num_threads(NUM_THREADS) private(x)
    for (i = 1; i < n - 1; i++) {
        x = a + i * step;
        #pragma omp critical
        integral += 2 * (f(x));
    }
    return integral * step / 2;
}

double IntegralAtomic(double a, double b, int n) {
    double step = (b - a) / n;
    double x, integral = 0;
    int i;
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (i = 1; i < n - 1; i++) {
        x = a + i * step;
        #pragma omp atomic
        integral += 2 * (f(x));
    }
    return integral * step / 2;
}


double IntegralReduction(double a, double b, int n) {
    double step = (b - a) / n;
    double x, integral = 0;
    int i;
#pragma omp parallel num_threads(NUM_THREADS)  private(x)

    #pragma omp for reduction(+:integral)
        for (i = 1; i < n - 1; i++) {
            x = a + i * step;
            integral += 2 * (f(x));
        }
    return integral * step / 2;
}

double IntegralLocks(double a, double b, int n) {
    double step = (b - a) / n;
    double x, integral = 0;
    int i;
    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel for num_threads(NUM_THREADS) private(x)
    for (i = 1; i < n - 1; i++)
    {
        x = a + i * step;
        omp_set_lock (&lock);
        integral += 2 * (f(x));
        omp_unset_lock (&lock);
    }
    omp_destroy_lock (&lock);

    return integral * step / 2;
}

int main() {
    double res;
    int n =N;
    double start;
    double end;
    double A[] = {0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10};
    double B[] = {0.0001, 0.001, 0.01, 0.1, 1, 10, 100};
    for (int i=0;i < 6; i++){
        double a = A[i];
        double b = B[i];
        printf("A=%f B=%f N=%d THREADS=%d \n", a, b, n,  NUM_THREADS);

        // classical integral computation
        start = omp_get_wtime();
        res = IntegralClassic(a, b, n);
        end = omp_get_wtime();
        printf("    Classic: result=%f time=%f sec. \n", res, end-start);


        // atomic integral computation
        start = omp_get_wtime();
        res = IntegralAtomic(a, b, n);
        end = omp_get_wtime();
        printf("    Atomic: result=%f time=%f sec. \n", res, end-start);


        //critical section integral computation
        start = omp_get_wtime();
        res = IntegralCritical(a, b, n);
        end = omp_get_wtime();
        printf("    Critical Section: result=%f time=%f sec.\n", res, end-start);


        //reduction integral computation
        start = omp_get_wtime();
        res = IntegralReduction(a, b, n);
        end = omp_get_wtime();
        printf("    Reduction: result=%f time=%f sec.\n", res, end-start);


        //Locks integral computation
        start = omp_get_wtime();
        res = IntegralLocks(a, b, n);
        end = omp_get_wtime();
        printf("    Locks: result=%f time=%f sec.\n", res, end-start);

    }


}