#include <iostream>
#include <vtkXMLGenericDataObjectReader.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkIdTypeArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkPolyDataWriter.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkImageData.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkDoubleArray.h>



template<typename n_t, typename n_to = double, int ncomp=3>
n_to *strip_component(n_t *aos, long nelem, long comp)
{
    n_to *soac = (n_to*)malloc(nelem*sizeof(n_to));

    aos += comp;

    #pragma omp parallel for
    for (long i = 0; i < nelem; ++i)
        soac[i] = aos[ncomp*i];

    return soac;

}

template<typename n_t>
n_t *compute_density(n_t *x, n_t *y, n_t *z, long np, n_t x0, n_t y0, n_t z0, n_t dx, n_t dy, n_t dz, long ncx, long ncy, long ncz)
{
    long ncxy = ncx*ncy;
    long mcx = ncx - 1;
    long mcy = ncy - 1;
    long mcz = ncz - 1;

    long *I = (long*)malloc(np*sizeof(long));
    #pragma omp parallel for
    for (long q = 0; q < np; ++q)
        I[q] = std::min(mcx, long((x[q] - x0)/dx));

    long *J = (long*)malloc(np*sizeof(long));
    #pragma omp parallel for
    for (long q = 0; q < np; ++q)
        J[q] = std::min(mcy, long((y[q] - y0)/dy));

    long *K = (long*)malloc(np*sizeof(long));
    #pragma omp parallel for
    for (long q = 0; q < np; ++q)
        K[q] = std::min(mcz, long((z[q] - z0)/dz));

    long nbytes = ncxy*ncz*sizeof(n_t);
    n_t *den = (n_t*)malloc(nbytes);
    memset(den, 0, nbytes);

    // TODO -- parallelize this loop
    for (long q = 0; q < np; ++q)
        den[K[q]*ncxy + J[q]*ncx + I[q]] += n_t(1);

    free(I);
    free(J);
    free(K);

    return den;
}

template<typename n_t>
int read_points_vtk(const std::string &fileName, n_t *&x, n_t *&y, n_t *&z, long &n)
{
    x = nullptr;
    y = nullptr;
    z = nullptr;
    n = 0;

    // read in point set
    vtkXMLGenericDataObjectReader *r = vtkXMLGenericDataObjectReader::New();
    r->SetFileName(fileName.c_str());
    r->Update();

    vtkUnstructuredGrid *ug =
      dynamic_cast<vtkUnstructuredGrid*>(r->GetOutput());
    if (!ug)
        return -1;

    // convert from aos to soa
    vtkFloatArray *pts = dynamic_cast<vtkFloatArray*>(ug->GetPoints()->GetData());
    n = pts->GetNumberOfTuples();

    float *ppts = pts->GetPointer(0);
    x = strip_component(ppts, n, 0);
    y = strip_component(ppts, n, 1);
    z = strip_component(ppts, n, 2);

    r->Delete();

    return 0;
}

template <typename n_t>
void min_max(n_t *v, long n, n_t &vmin, n_t &vmax)
{
    vmin = std::numeric_limits<n_t>::max();
    #pragma omp parallel for reduction(min:vmin)
    for (long i = 0; i < n; ++i)
        vmin = std::min(vmin, v[i]);

    vmax = std::numeric_limits<n_t>::lowest();
    #pragma omp parallel for reduction(max:vmax)
    for (long i = 0; i < n; ++i)
        vmax = std::max(vmax, v[i]);
}

template<typename n_t>
void compute_bounds(n_t *x, n_t *y, n_t *z, long n, double *bds)
{
    min_max(x, n, bds[0], bds[1]);
    min_max(y, n, bds[2], bds[3]);
    min_max(z, n, bds[4], bds[5]);
}

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        cerr << "density [in file] [out file] [npx] [ny] [nz]" << endl;
        return -1;
    }

    std::string fileName = argv[1];
    std::string outFileName = argv[2];
    int npx[3] = {atoi(argv[3]), atoi(argv[4]), atoi(argv[5])};
    int ncx[3] = {npx[0] - 1, npx[1] - 1, npx[2] - 1};

    double *x = nullptr;
    double *y = nullptr;
    double *z = nullptr;
    long n = 0;

    // read in point set
    if (read_points_vtk(fileName, x, y, z, n))
    {
        cerr << "failed to read \"" << fileName << "\"" << endl;
        return -1;
    }

    // construct output volume
    double bounds[6] = {0.0};
    compute_bounds(x, y, z, n, bounds);

    double lx[3] = {bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]};
    double dx[3] = {lx[0]/ncx[0],  lx[1]/ncx[1], lx[2]/ncx[2]};
    double x0[3] = {bounds[0], bounds[2], bounds[4]};

    vtkImageData *im = vtkImageData::New();
    im->SetDimensions(ncx);
    im->SetSpacing(dx);
    im->SetOrigin(x0[0] + dx[0], x0[1] + dx[1], x0[2] + dx[2]);

    // compute density
    double *den = compute_density(x, y, z, n, x0[0], x0[1], x0[2], dx[0], dx[1], dx[2], ncx[0], ncx[1], ncx[2]);

    // deallocate points
    free(x);
    free(y);
    free(z);

    // pass density calc into vtk
    vtkDoubleArray *denArray = vtkDoubleArray::New();
    denArray->SetName("den");
    denArray->SetArray(den, ncx[0]*ncx[1]*ncx[2], 0);

    im->GetPointData()->AddArray(denArray);
    denArray->Delete();

    // write the volume
    vtkStructuredPointsWriter *writer = vtkStructuredPointsWriter::New();
    writer->SetFileName(outFileName.c_str());
    writer->SetInputData(im);
    writer->SetFileTypeToBinary();
    writer->Write();

    im->Delete();
    writer->Delete();

    return 0;
}
