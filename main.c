#include "pde.h"
double f1(double x, double y)
{
    return -20. * x * 10. * y - 100.;
//    return 0.;
}
int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    pde(argc, argv, f1);
    MPI_Finalize();
    return 0;
}