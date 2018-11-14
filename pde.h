#ifndef PDE_H
#define PDE_H
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

const double eps = 0.01;
const int save = 0;
/**
 * f(x,y)=0
 * 100-200x, y=0
 * 100-200y, x=0
 * −100+200x, y=1
 * −100+200y, x=1
 */

void paralleled_method(int N, double(*func)(double, double), const char *path, int save)
{
    int my_rank, num_procs, strip_size, strip_size1, strip_size2, i, j, r, c;
    int a;
    double dmax;
    double *u, *f, dm, dm_inner, temp, d;
    double h = 1. / (N - 1);
    double tstart;
    int k = 0;
    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    strip_size1 = N / num_procs;
    strip_size2 =  N - (num_procs - 1) * strip_size1;
    strip_size = my_rank == num_procs - 1 ? strip_size2 : strip_size1;
    if (my_rank == 0) {
        u = (double *)malloc(sizeof(double) * N * N);
        a = my_rank != num_procs - 1;
        f = (double *)malloc(sizeof(double) * N * (strip_size + a));
        for (i = 0; i < N * (strip_size + a); i++) {
            c = i % N;
            r = i / N;
            f[i] = func(c * h, r * h);
            if (i < N) {
                u[i] = 100. - 200. * h * c;
            } else if (i >= N * (N - 1)) {
                u[i] = -100. + 200. * h * c;
            } else if (c == 0) {
                u[i] = 100. - 200. * h * r;
            } else if (c == N - 1) {
                u[i] = -100 + 200 * h * r;
            } else {
                u[i] = 0;
            }

        }
    } else if(my_rank == num_procs - 1) {
        u = (double *)malloc(sizeof(double) * N * (strip_size + 1));
        f = (double *)malloc(sizeof(double) * N * (strip_size + 1));
        for (i = 0; i < N * (strip_size + 1); i++) {
            c = i % N;
            r = i / N + strip_size1 * my_rank - 1;
            f[i] = func(c * h, r * h);
            if (i >= N * strip_size) {
                u[i] = -100. + 200. * h * c;
            } else if (c == 0) {
                u[i] = 100. - 200. * h * r;
            } else if (c == N - 1) {
                u[i] = -100 + 200 * h * r;
            } else {
                u[i] = 0;
            }

        }
    } else {
        u = (double *)malloc(sizeof(double) * N * (strip_size + 2));
        f = (double *)malloc(sizeof(double) * N * (strip_size + 2));
        for (i = 0; i < N * (strip_size + 2); i++) {
            c = i % N;
            r = i / N + strip_size * my_rank - 1;
            f[i] = func(c * h, r * h);
            if (c == 0) {
                u[i] = 100. - 200. * h * r;
            } else if (c == N - 1) {
                u[i] = -100 + 200 * h * r;
            } else {
                u[i] = 0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    tstart = MPI_Wtime();
    do {
        k += 1;
        if (my_rank % 2) {
            a = my_rank != num_procs - 1;
            dm = -1;
#pragma omp parallel for private(j, i, temp, dm_inner, d) shared(u, dm, f, a, N, strip_size, h)
            for (j = 1; j < strip_size + a; j += 2) {
                dm_inner = -1;
                for (i = 1; i < N - 1; i++) {
                    temp = u[j * N + i];
                    u[j * N + i] = 0.25 * (u[j * N + i - 1] + u[j * N + i + 1]
                                           + u[(j - 1) * N + i] + u[(j + 1) * N + i] - h * h * f[j * N + i]);
                    d = fabs(u[j * N + i] - temp);
                    if (d > dm_inner) {
                        dm_inner = d;
                    }
                }
                omp_set_lock(&dmax_lock);
                if (dm_inner > dm) {
                    dm = dm_inner;
                }
                omp_unset_lock(&dmax_lock);
            }

#pragma omp parallel for private(j, i, temp, dm_inner, d) shared(u, dm, f, a, N, strip_size, h)
            for (j = 2; j < strip_size + a; j += 2) {
                dm_inner = -1;
                for (i = 1; i < N - 1; i++) {
                    temp = u[j * N + i];
                    u[j * N + i] = 0.25 * (u[j * N + i - 1] + u[j * N + i + 1]
                                           + u[(j - 1) * N + i] + u[(j + 1) * N + i] - h * h * f[j * N + i]);
                    d = fabs(u[j * N + i] - temp);
                    if (d > dm_inner) {
                        dm_inner = d;
                    }
                }
                omp_set_lock(&dmax_lock);
                if (dm_inner > dm) {
                    dm = dm_inner;
                }
                omp_unset_lock(&dmax_lock);
            }

            if (my_rank != num_procs - 1) {
                MPI_Sendrecv(&u[(strip_size + a - 1) * N], N, MPI_DOUBLE, my_rank + 1, 0, &u[(strip_size + a) * N], N, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                MPI_Send(&u[(strip_size + a - 1) * N], N, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
//                MPI_Recv(&u[(strip_size + a) * N], N, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            }
            MPI_Sendrecv(&u[N], N, MPI_DOUBLE, my_rank - 1, 1, u, N, MPI_DOUBLE, my_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            MPI_Send(&u[N], N, MPI_DOUBLE, my_rank - 1, 1, MPI_COMM_WORLD);
//            MPI_Recv(u, N, MPI_DOUBLE, my_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            a = !(my_rank == 0 || my_rank == num_procs - 1);
            a = (my_rank == 0 && my_rank == num_procs - 1) ? -1 : a;


            dm = -1;
#pragma omp parallel for private(j, i, temp, dm_inner, d) shared(u, dm, f, a, N, strip_size, h)
            for (j = 1; j < strip_size + a; j += 2) {
                dm_inner = -1;
                for (i = 1; i < N - 1; i++) {
                    temp = u[j * N + i];
                    u[j * N + i] = 0.25 * (u[j * N + i - 1] + u[j * N + i + 1]
                                           + u[(j - 1) * N + i] + u[(j + 1) * N + i] - h * h * f[j * N + i]);
                    d = fabs(u[j * N + i] - temp);
                    if (d > dm_inner) {
                        dm_inner = d;
                    }
                }
                omp_set_lock(&dmax_lock);
                if (dm_inner > dm) {
                    dm = dm_inner;
                }
                omp_unset_lock(&dmax_lock);
            }
#pragma omp parallel for private(j, i, temp, dm_inner, d) shared(u, dm, f, a, N, strip_size, h)
            for (j = 2; j < strip_size + a; j += 2) {
                dm_inner = -1;
                for (i = 1; i < N - 1; i++) {
                    temp = u[j * N + i];
                    u[j * N + i] = 0.25 * (u[j * N + i - 1] + u[j * N + i + 1]
                                           + u[(j - 1) * N + i] + u[(j + 1) * N + i] - h * h * f[j * N + i]);
                    d = fabs(u[j * N + i] - temp);
                    if (d > dm_inner) {
                        dm_inner = d;
                    }
                }
                omp_set_lock(&dmax_lock);
                if (dm_inner > dm) {
                    dm = dm_inner;
                }
                omp_unset_lock(&dmax_lock);
            }
            if (my_rank != 0) {
                MPI_Sendrecv(&u[N], N, MPI_DOUBLE, my_rank - 1, 0, u, N, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                MPI_Recv(u, N, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
//            if (my_rank != num_procs - 1) {
//                MPI_Recv(&u[(strip_size + a) * N], N, MPI_DOUBLE, my_rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            }
//            if (my_rank != 0) {
//                MPI_Send(&u[N], N, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
//            }
            if (my_rank != num_procs - 1) {
                MPI_Sendrecv(&u[(strip_size + a - 1) * N], N, MPI_DOUBLE, my_rank + 1, 1, &u[(strip_size + a) * N], N, MPI_DOUBLE, my_rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Reduce(&dm, &dmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&dmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } while (dmax > eps);
    if (my_rank == 0) {
        for (i = 1; i < num_procs; i++) {
            if (i == num_procs - 1) {
                MPI_Recv(&u[N * i * strip_size], N * strip_size2, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(&u[N * i * strip_size], N * strip_size1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        MPI_Send(&u[N], N * strip_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {
        printf("Duration: %.3fms\n", 1e3 * (MPI_Wtime() - tstart));
        printf("Iterations: %d\n", k);
        if (save) {
            FILE *fp = fopen(path, "wb");
            fwrite(u, sizeof(double), N * N, fp);
            fclose(fp);
        }
    }
}
void thread_method(int N, double(*func)(double, double), const char *path, int save)
{
    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);
    int i, j, k = 0;
    double dmax, dm, d;
    double temp;
    double tstart;
    double *u = (double *)malloc(sizeof(double) * N * N);
    double *f = (double *)malloc(sizeof(double) * N * N);
    double h = 1. / (N - 1);
#pragma omp parallel for private(i) shared(u)
    for (i = 0; i < N * N; i++) {
        u[i] = 0.;
        f[i] = func(i % N * h, i / N * h);
    }
#pragma omp parallel for private(i) shared(u)
    for (i = 0; i < N; i++) {
        u[i] = 100. - 200. * h * i;
        u[(N - 1) * N + i] = -100. + 200. * h * i;
    }
#pragma omp parallel for private(j) shared(u)
    for (j = 0; j < N; j++) {
        u[j * N] = 100. - 200. * h * j;
        u[j * N + N - 1] = -100 + 200 * h * j;
    }
    tstart = MPI_Wtime();
    do {
        k += 1;
        dmax = -1.;



#pragma omp parallel for private(j, i, temp, dm, d) shared(u, dmax, f, N)
        for (j = 1; j < N - 1; j+=2) {
            dm = -1;
            for (i = 1; i < N - 1; i++) {
                temp = u[j * N + i];
                u[j * N + i] = 0.25 * (u[j * N + i - 1] + u[j * N + i + 1]
                                       + u[(j - 1) * N + i] + u[(j + 1) * N + i] - h * h * f[j * N + i]);
                d = fabs(u[j * N + i] - temp);
                if (d > dm) {
                    dm = d;
                }

            }
            omp_set_lock(&dmax_lock);
            if (dm > dmax) {
                dmax = dm;
            }
            omp_unset_lock(&dmax_lock);
        }
#pragma omp parallel for private(j, i, temp, dm, d) shared(u, dmax, f, N)
        for (j = 2; j < N - 1; j+=2) {
            dm = -1;
            for (i = 1; i < N - 1; i++) {
                temp = u[j * N + i];
                u[j * N + i] = 0.25 * (u[j * N + i - 1] + u[j * N + i + 1]
                                       + u[(j - 1) * N + i] + u[(j + 1) * N + i] - h * h * f[j * N + i]);
                d = fabs(u[j * N + i] - temp);
                if (d > dm) {
                    dm = d;
                }

            }
            omp_set_lock(&dmax_lock);
            if (dm > dmax) {
                dmax = dm;
            }
            omp_unset_lock(&dmax_lock);
        }


    } while (dmax > eps);
    printf("Duration: %.3fms\n", 1e3*(MPI_Wtime() - tstart));
    printf("Iterations: %d\n", k);
    if (save) {
        FILE *fp = fopen(path, "wb");
        fwrite(u, sizeof(double), N * N, fp);
        fclose(fp);
    }
}


void sequential_method(int N, double(*func)(double, double), const char *path, int save)
{
    int i, j, k = 0;
    double dmax, dm, d;
    double temp;
    double tstart;
    double *u = (double *)malloc(sizeof(double) * N * N);
    double *f = (double *)malloc(sizeof(double) * N * N);
    double h = 1. / (N - 1);
    for (i = 0; i < N * N; i++) {
        u[i] = 0.;
        f[i] = func(i % N * h, i / N * h);
    }
    for (i = 0; i < N; i++) {
        u[i] = 100. - 200. * h * i;
        u[(N - 1) * N + i] = -100. + 200. * h * i;
    }
    for (j = 0; j < N; j++) {
        u[j * N] = 100. - 200. * h * j;
        u[j * N + N - 1] = -100 + 200 * h * j;
    }
    tstart = MPI_Wtime();
    do {
        k += 1;
        dmax = -1.;
        for (j = 1; j < N - 1; j+=1) {
            dm = -1;
            for (i = 1; i < N - 1; i++) {
                temp = u[j * N + i];
                u[j * N + i] = 0.25 * (u[j * N + i - 1] + u[j * N + i + 1]
                                       + u[(j - 1) * N + i] + u[(j + 1) * N + i] - h * h * f[j * N + i]);
                d = fabs(u[j * N + i] - temp);
                if (d > dm) {
                    dm = d;
                }

            }
            if (dm > dmax) {
                dmax = dm;
            }
        }
    } while (dmax > eps);
    printf("Duration: %.3fms\n", 1e3*(MPI_Wtime() - tstart));
    printf("Iterations: %d\n", k);
    if (save) {
        FILE *fp = fopen(path, "wb");
        fwrite(u, sizeof(double), N * N, fp);
        fclose(fp);
    }
}

void pde(int argc, char *argv[], double(*func)(double, double))
{


    const char *path = "u.data";
    int opt = 0;
    int N = 100;
    int t = 1;
//    int t = 0;
    int method = 0;
    const char *short_opts = "o:g:t:m:";
    const struct option long_opts[] = {
            {"grids", required_argument, NULL, 'g'},
            {"output", required_argument, NULL, 'o'},
            {"threads", required_argument, NULL, 't'},
            {"method", required_argument, NULL, 'm'},
            {0, 0, 0, 0}
    };

    while ( (opt = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1 ) {
        switch (opt) {
            case 'o':
                path = optarg;
                break;
            case 'g':
                N = atoi(optarg);
                break;
            case 't':
                t = atoi(optarg);
                break;
            case 'm':
                method = atoi(optarg);
                break;
            default:
                exit(1);
        }
    }
    omp_set_num_threads(t);
    switch (method) {
        case 0:
            paralleled_method(N, func, path, save);
            break;
        case 1:
            thread_method(N, func, path, save);
            break;
        default:
            sequential_method(N, func, path, save);
            break;
    }


}



#endif
