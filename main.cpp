#include <iostream>
#include "Eigen/Eigen"
#include <cmath>
#include <numbers>
#include <iomanip>

using namespace std;
using namespace Eigen;

bool SystemPALU(const Matrix2d& A, double& detA, double& condA, double& err_relA, const Vector2d& b)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d singularValuesA = svd.singularValues();
    condA = singularValuesA.maxCoeff() / singularValuesA.minCoeff();
    detA = A.determinant();

    if( singularValuesA.minCoeff() < 1e-16)
    {
        err_relA = -1;
        return false;
    }

    unsigned int n = A.rows();
    Vector2d exactSolution = Vector2d::Ones(n) * -1;
    Vector2d x = A.fullPivLu().solve(b);
    err_relA = (exactSolution - x).norm() / exactSolution.norm();
    return true;
}

bool SystemQR(const Matrix2d& A, double& detA, double& condA, double& err_relA, const Vector2d& b)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d singularValuesA = svd.singularValues();
    condA = singularValuesA.maxCoeff() / singularValuesA.minCoeff();
    detA = A.determinant();

    if( singularValuesA.minCoeff() < 1e-16)
    {
        err_relA = -1;
        return false;
    }

    unsigned int n = A.rows();
    Vector2d exactSolution = Vector2d::Ones(n) * -1;
    Vector2d x = A.fullPivHouseholderQr().solve(b);
    err_relA = (exactSolution - x).norm() / exactSolution.norm();
    return true;
}

int main()
{
    Matrix2d A1 = (Matrix2d() << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01,-9.992887623566787e-01).finished();
    Vector2d b1 = (Vector2d() << -5.169911863249772e-01, 1.672384680188350e-01).finished();
    Matrix2d A2 = (Matrix2d() << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,-8.324762492991313e-01).finished();
    Vector2d b2 = (Vector2d() << -6.394645785530173e-04, 4.259549612877223e-04).finished();
    Matrix2d A3 = (Matrix2d() << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,-8.320502947645361e-01).finished();
    Vector2d b3 = (Vector2d() << -6.400391328043042e-10, 4.266924591433963e-10).finished();

    cout << scientific << "A1:" << endl << A1 << endl << endl;
    cout << scientific << "b1:" << endl << b1 << endl << endl;
    cout << scientific << "A2:" << endl << A2 << endl << endl;
    cout << scientific << "b2:" << endl << b2 << endl << endl;
    cout << scientific << "A3:" << endl << A3 << endl << endl;
    cout << scientific << "b3:" << endl << b3 << endl << endl;


    cout << " Resolution PALU:" << endl;

    double detA1, condA1, err_relA1;
    if(SystemPALU(A1, detA1, condA1, err_relA1,b1))
        cout << "A1" << " Relative Error: "<< err_relA1<< endl;
    else
        cout << "A1" << " (Matrix is singular)"<< endl;

    double detA2, condA2, err_relA2;
    if(SystemPALU(A2, detA2, condA2, err_relA1,b2))
        cout << "A2" << " Relative Error: "<< err_relA2<< endl;
    else
        cout << "A2" << " (Matrix is singular)"<< endl;

    double detA3, condA3, err_relA3;
    if(SystemPALU(A3, detA3, condA3, err_relA3,b3))
        cout << "A3"<< " Relative Error: "<< err_relA3<< endl;
    else
        cout << "A3" << " (Matrix is singular)"<< endl;

    cout<<endl;

    cout << " Resolution QR:" << endl;

    if(SystemQR(A1, detA1, condA1, err_relA1,b1))
        cout << "A1" << " Relative Error: "<< err_relA1<< endl;
    else
        cout << "A1" << " Matrix is singular"<< endl;

    if(SystemQR(A2, detA2, condA2, err_relA2,b2))
        cout << "A2" << " Relative Error: "<< err_relA2<< endl;
    else
        cout << "A2" << " Matrix is singular"<< endl;

    if(SystemQR(A3, detA3, condA3, err_relA3,b3))
        cout << "A3" << " Relative Error: "<< err_relA3<<endl;
    else
        cout << "A3" << " Matrix is singular"<< endl;

}