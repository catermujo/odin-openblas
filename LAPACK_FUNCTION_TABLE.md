# LAPACK Function Mapping

## Signature Naming Convention

| Prefix     | Type                    | Example Functions                                 |
|------------|-------------------------|---------------------------------------------------|
| dns_       | Dense matrices          | dns_solve, dns_eigen_symmetric, dns_svd_dc        |
| band_      | Banded matrices         | band_cholesky, band_eigen, band_solve             |
| tri_       | Triangular matrices     | tri_solve, tri_invert, tri_refine                 |
| trid_      | Tridiagonal matrices    | trid_eigen_mrrr, trid_solve, trid_norm            |
| pack_sym_  | Packed symmetric        | pack_sym_eigen, pack_sym_solve, pack_sym_cholesky |
| pack_herm_ | Packed Hermitian        | pack_herm_solve, pack_herm_condition              |
| pack_tri_  | Packed triangular       | pack_tri_make, pack_tri_from_dns                  |
| bidi_      | Bidiagonal              | bidi_svd, bidi_reduce_from_dns                    |
| rfp_       | Rectangular Full Packed | rfp_cholesky, rfp_tri_solve                       |
| v_         | Vector operations       | v_householder_generate, v_givens_apply            |

Key Accomplishments

Procedures follow {storage_type}_{operation}_{method/variant}_{precision}

## LAPACK Function Reference Table

| Function | Variants | Matrix Type | Description | Target File |
|----------|----------|-------------|-------------|-------------|
| bbcsd | c,d,s,z | Bidiagonal | Block bidiagonal cosine-sine decomposition | bidiagonal.odin |
| bdsdc | d,s | Bidiagonal | SVD of bidiagonal matrix (divide-and-conquer) | bidiagonal_svd.odin |
| bdsqr | c,d,s,z | Bidiagonal | SVD of bidiagonal matrix | bidiagonal_svd.odin |
| bdsvdx | d,s | Bidiagonal | Selected singular values of bidiagonal matrix | bidiagonal_svd.odin |
| cgesv | z | Dense | Solve general linear system AX=B (complex single) | dense_linear.odin |
| cposv | z | Dense | Solve positive definite linear system (complex single) | dense_positive_definite.odin |
| disna | d,s | Utility | Reciprocal condition numbers for eigenvalues | auxiliary.odin |
| gbbrd | c,d,s,z | Banded | Reduce to bidiagonal form | banded_decomposition.odin |
| gbcon | c,d,s,z | Banded | Condition number estimation for banded matrix | banded_linear.odin |
| gbequ | c,d,s,z | Banded | Row and column scaling for banded matrix | banded_linear.odin |
| gbequb | c,d,s,z | Banded | Improved row/column scaling for banded matrix | banded_linear.odin |
| gbrfs | c,d,s,z | Banded | Iterative refinement for banded linear system | banded_linear.odin |
| gbrfsx | c,d,s,z | Banded | Expert iterative refinement for banded system | banded_linear.odin |
| gbsv | c,d,s,z | Banded | General banded linear system solver | banded_linear.odin |
| gbsvx | c,d,s,z | Banded | Expert solver for general banded linear system | banded_linear.odin |
| gbsvxx | c,d,s,z | Banded | Extra expert solver for banded linear system | banded_linear.odin |
| gbtrf | c,d,s,z | Banded | LU factorization of banded matrix | banded_decomposition.odin |
| gbtrs | c,d,s,z | Banded | Solve banded system using LU factorization | banded_linear.odin |
| gebak | c,d,s,z | Dense | Back-transform eigenvectors after balancing | dense_eigenvalues.odin |
| gebal | c,d,s,z | Dense | Balance matrix for eigenvalue computation | dense_eigenvalues.odin |
| gebrd | c,d,s,z | Dense | Reduce general matrix to bidiagonal form | dense_factorization.odin |
| gecon | c,d,s,z | Dense | Condition number estimation | dense_linear.odin |
| gedmd | c,d,s,z | Dense | Dynamic mode decomposition | dmd.odin |
| gedmdq | c,d,s,z | Dense | Dynamic mode decomposition with QR | dmd.odin |
| geequ | c,d,s,z | Dense | Row and column scaling factors | dense_linear.odin |
| geequb | c,d,s,z | Dense | Improved row/column scaling for general matrix | dense_linear.odin |
| gees | c,d,s,z | Dense | Schur factorization | dense_eigenvalues.odin |
| geesx | c,d,s,z | Dense | Schur factorization (expert) | dense_eigenvalues.odin |
| geev | c,d,s,z | Dense | General eigenvalue problem | dense_eigenvalues.odin |
| geevx | c,d,s,z | Dense | General eigenvalue problem (expert) | dense_eigenvalues.odin |
| gehrd | c,d,s,z | Dense | Reduce general matrix to Hessenberg form | dense_factorization.odin |
| gejsv | c,d,s,z | Dense | Jacobi SVD with high accuracy | dense_svd.odin |
| gelq | c,d,s,z | Orthogonal | LQ factorization (short-wide recursive) | orthogonal_transformations.odin |
| gelqf | c,d,s,z | Orthogonal | LQ factorization | orthogonal_transformations.odin |
| gelqt | c,d,s,z | Orthogonal | LQ factorization using compact WY representation | orthogonal_transformations.odin |
| gels | c,d,s,z | Dense | Linear least squares problem | dense_least_squares.odin |
| gelsd | c,d,s,z | Dense | Linear least squares problem (divide-and-conquer SVD) | dense_least_squares.odin |
| gelss | c,d,s,z | Dense | Linear least squares problem (SVD) | dense_least_squares.odin |
| gelst | c,d,s,z | Dense | Solve least squares using tall-skinny QR | dense_least_squares.odin |
| gelsy | c,d,s,z | Dense | Linear least squares problem (rank-revealing QR) | dense_least_squares.odin |
| gemlq | c,d,s,z | Orthogonal | Multiply by Q from LQ factorization (recursive) | orthogonal_transformations.odin |
| gemqr | c,d,s,z | Orthogonal | Multiply by Q from QR factorization (recursive) | orthogonal_transformations.odin |
| gemqrt | c,d,s,z | Orthogonal | Multiply by Q using compact WY representation | orthogonal_transformations.odin |
| geqlf | c,d,s,z | Orthogonal | QL factorization | orthogonal_transformations.odin |
| geqpf | c,d,s,z | Dense | QR factorization with column pivoting | dense_factorization.odin |
| geqr | c,d,s,z | Orthogonal | QR factorization (tall-skinny recursive) | orthogonal_transformations.odin |
| geqrf | c,d,s,z | Orthogonal | QR factorization | orthogonal_transformations.odin |
| geqrfp | c,d,s,z | Orthogonal | QR factorization with positive diagonal R | orthogonal_transformations.odin |
| geqrt | c,d,s,z | Orthogonal | QR factorization using compact WY representation | orthogonal_transformations.odin |
| gerfs | c,d,s,z | Dense | Iterative refinement | dense_linear.odin |
| gerfsx | c,d,s,z | Dense | Expert iterative refinement for general system | dense_linear.odin |
| gerqf | c,d,s,z | Orthogonal | RQ factorization | orthogonal_transformations.odin |
| gesdd | c,d,s,z | Dense | Singular value decomposition (divide-and-conquer) | dense_svd.odin |
| gesvd | c,d,s,z | Dense | Singular value decomposition | dense_svd.odin |
| gesvdq | c,d,s,z | Dense | Singular value decomposition (QR preconditioner) | dense_svd.odin |
| gesvdx | c,d,s,z | Dense | Singular value decomposition (selected) | dense_svd.odin |
| gesvj | c,d,s,z | Dense | LAPACK function gesvj | dense_svd.odin |
| gesvx | c,d,s,z | Dense | Expert solver for general linear system | dense_linear.odin |
| gesvxx | c,d,s,z | Dense | Extra expert solver for general linear system | dense_linear.odin |
| getri | c,d,s,z | Dense | Matrix inversion after LU factorization | dense_linear.odin |
| getsls | c,d,s,z | Dense | Solve least squares using tall-skinny LQ | dense_least_squares.odin |
| getsqrhrt | c,d,s,z | Orthogonal | QR factorization with Householder reconstruction | orthogonal_transformations.odin |
| ggbak | c,d,s,z | Dense | Back-transform eigenvectors after generalized balancing | dense_eigenvalues.odin |
| ggbal | c,d,s,z | Dense | Balance matrix pair for generalized eigenvalues | dense_eigenvalues.odin |
| gges | c,d,s,z | Dense | Generalized Schur factorization | dense_eigenvalues.odin |
| ggesx | c,d,s,z | Dense | Generalized Schur factorization (expert) | dense_eigenvalues.odin |
| ggev | c,d,s,z | Dense | Generalized eigenvalue problem | dense_eigenvalues.odin |
| ggevx | c,d,s,z | Dense | Generalized eigenvalue problem (expert) | dense_eigenvalues.odin |
| ggglm | c,d,s,z | Dense | General Gauss-Markov linear model | dense_least_squares.odin |
| gghrd | c,d,s,z | Dense | Reduce matrix pair to generalized Hessenberg form | dense_factorization.odin |
| gglse | c,d,s,z | Dense | Linear equality-constrained least squares | dense_least_squares.odin |
| ggqrf | c,d,s,z | Orthogonal | Generalized QR factorization of matrix pair | orthogonal_transformations.odin |
| ggrqf | c,d,s,z | Orthogonal | Generalized RQ factorization of matrix pair | orthogonal_transformations.odin |
| ggsvd | c,d,s,z | Dense | Generalized singular value decomposition | dense_svd.odin |
| ggsvp | c,d,s,z | Dense | Preprocessing for generalized SVD | dense_svd.odin |
| gtcon | c,d,s,z | Tridiagonal | Condition number of tridiagonal matrix | tridiagonal_linear.odin |
| gtrfs | c,d,s,z | Tridiagonal | Iterative refinement for tridiagonal system | tridiagonal_linear.odin |
| gtsv | c,d,s,z | Tridiagonal | General tridiagonal linear system solver | tridiagonal_linear.odin |
| gtsvx | c,d,s,z | Tridiagonal | Expert solver for general tridiagonal system | tridiagonal_linear.odin |
| gttrf | c,d,s,z | Tridiagonal | LU factorization of tridiagonal matrix | tridiagonal.odin |
| gttrs | c,d,s,z | Tridiagonal | Solve tridiagonal system after factorization | tridiagonal_linear.odin |
| hbev | c,z | Banded | Eigenvalues/vectors of Hermitian banded matrix | banded_eigenvalues.odin |
| hbevd | c,z | Banded | Hermitian banded eigenvalues (divide-and-conquer) | banded_eigenvalues.odin |
| hbevx | c,z | Banded | Selected eigenvalues of Hermitian banded matrix | banded_eigenvalues.odin |
| hbgst | c,z | Banded | Reduce generalized Hermitian banded to standard form | banded.odin |
| hbgv | c,z | Banded | Generalized Hermitian banded eigenvalue problem | banded_eigenvalues.odin |
| hbgvd | c,z | Banded | Generalized Hermitian banded eigenvalues (divide-and-conquer) | banded_eigenvalues.odin |
| hbgvx | c,z | Banded | Selected generalized Hermitian banded eigenvalues | banded_eigenvalues.odin |
| hbtrd | c,z | Banded | Reduce Hermitian banded to tridiagonal form | banded_decomposition.odin |
| hecon | c,z | Dense | Condition number estimation for Hermitian matrix | dense_hermitian_linear.odin |
| heequb | c,z | Dense | Improved scaling for Hermitian matrix | dense_hermitian_linear.odin |
| heev | c,z | Dense | Hermitian eigenvalue problem | dense_hermitian_eigenvalues.odin |
| heevd | c,z | Dense | Hermitian eigenvalue problem (divide-and-conquer) | dense_hermitian_eigenvalues.odin |
| heevr | c,z | Dense | Hermitian eigenvalue problem (RRR) | dense_hermitian_eigenvalues.odin |
| heevx | c,z | Dense | Hermitian eigenvalue problem (selected) | dense_hermitian_eigenvalues.odin |
| hegst | c,z | Dense | Reduce generalized Hermitian eigenvalue problem to standard form | dense_hermitian_eigenvalues.odin |
| hegv | c,z | Dense | Generalized Hermitian eigenvalue problem | dense_hermitian_eigenvalues.odin |
| hegvd | c,z | Dense | Generalized Hermitian eigenvalue problem (divide-and-conquer) | dense_hermitian_eigenvalues.odin |
| hegvx | c,z | Dense | Generalized Hermitian eigenvalue problem (selected) | dense_hermitian_eigenvalues.odin |
| herfs | c,z | Dense | Iterative refinement for Hermitian system | dense_hermitian_linear.odin |
| herfsx | c,z | Dense | Expert iterative refinement for Hermitian system | dense_hermitian_linear.odin |
| hesv | c,z | Dense | Hermitian indefinite linear system solver | dense_hermitian_linear.odin |
| hesvx | c,z | Dense | Expert solver for Hermitian indefinite system | dense_hermitian_linear.odin |
| hesvxx | c,z | Dense | Extra expert solver for Hermitian system | dense_hermitian_linear.odin |
| heswapr | c,z | Dense | Apply row/column swaps to Hermitian matrix | dense_hermitian_linear.odin |
| hetrd | c,z | Dense | Reduce Hermitian matrix to tridiagonal form | dense_factorization.odin |
| hetrf | c,z | Dense | Bunch-Kaufman factorization of Hermitian matrix | dense_factorization.odin |
| hetri | c,z | Dense | Invert Hermitian matrix using factorization | dense_hermitian_linear.odin |
| hetrs | c,z | Dense | Solve Hermitian system using factorization | dense_hermitian_linear.odin |
| hfrk | c,z | RFP | Hermitian rank-k update in RFP format | rfp.odin |
| hgeqz | c,d,s,z | Dense | Generalized Schur form using QZ algorithm | dense_eigenvalues.odin |
| hpcon | c,z | Packed | Condition number of packed Hermitian matrix | packed_hermitian.odin |
| hpev | c,z | Packed | Eigenvalues/vectors of packed Hermitian matrix | packed_symmetric_eigenvalues.odin |
| hpevd | c,z | Packed | Packed Hermitian eigenvalues (divide-and-conquer) | packed_symmetric_eigenvalues.odin |
| hpevx | c,z | Packed | Selected eigenvalues of packed Hermitian matrix | packed_symmetric_eigenvalues.odin |
| hpgst | c,z | Packed | Reduce generalized packed Hermitian to standard form | packed_hermitian.odin |
| hpgv | c,z | Packed | Generalized packed Hermitian eigenvalue problem | packed_symmetric_eigenvalues.odin |
| hpgvd | c,z | Packed | Generalized packed Hermitian eigenvalues (divide-and-conquer) | packed_symmetric_eigenvalues.odin |
| hpgvx | c,z | Packed | Selected generalized packed Hermitian eigenvalues | packed_symmetric_eigenvalues.odin |
| hprfs | c,z | Packed | Iterative refinement for packed Hermitian system | packed_hermitian.odin |
| hpsv | c,z | Packed | Hermitian indefinite packed linear system solver | packed_hermitian.odin |
| hpsvx | c,z | Packed | Expert solver for packed Hermitian system | packed_hermitian.odin |
| hptrd | c,z | Packed | Reduce packed Hermitian to tridiagonal form | packed_hermitian.odin |
| hptrf | c,z | Packed | Bunch-Kaufman factorization of packed Hermitian matrix | packed_hermitian.odin |
| hptri | c,z | Packed | Invert packed Hermitian matrix using factorization | packed_hermitian.odin |
| hptrs | c,z | Packed | Solve packed Hermitian system using factorization | packed_hermitian.odin |
| hsein | c,d,s,z | Dense | Eigenvectors of Hessenberg matrix by inverse iteration | dense_eigenvalues.odin |
| hseqr | c,d,s,z | Dense | Eigenvalues/Schur form of Hessenberg matrix | dense_eigenvalues.odin |
| lacgv | c,z | Utility | Conjugate complex vector | auxiliary.odin |
| lacpy | c,d,s,z | Utility | Copy matrix or submatrix | auxiliary.odin |
| lacrm | c,z | Utility | Complex matrix times real matrix | auxiliary.odin |
| lagge | c,d,s,z | Utility | Generate general test matrix | test_matrix_generation.odin |
| laghe | c,z | Utility | Generate Hermitian test matrix | test_matrix_generation.odin |
| lagsy | c,d,s,z | Utility | Generate symmetric test matrix | test_matrix_generation.odin |
| langb | c,d,s,z | Utility | General banded matrix norm | matrix_norms.odin |
| lange | c,d,s,z | Utility | Matrix norm | matrix_norms.odin |
| langt | c,d,s,z | Utility | General tridiagonal matrix norm | matrix_norms.odin |
| lanhb | c,z | Utility | Hermitian banded matrix norm | matrix_norms.odin |
| lanhe | c,z | Utility | Hermitian matrix norm | matrix_norms.odin |
| lanhp | c,z | Utility | Hermitian packed matrix norm | matrix_norms.odin |
| lanhs | c,d,s,z | Utility | Hessenberg matrix norm | matrix_norms.odin |
| lanht | c,z | Utility | Hermitian tridiagonal matrix norm | matrix_norms.odin |
| lansb | c,d,s,z | Utility | Symmetric banded matrix norm | matrix_norms.odin |
| lansp | c,d,s,z | Utility | Symmetric packed matrix norm | matrix_norms.odin |
| lanst | d,s | Utility | Symmetric tridiagonal matrix norm | matrix_norms.odin |
| lansy | c,d,s,z | Utility | Symmetric matrix norm | matrix_norms.odin |
| lantb | c,d,s,z | Utility | Triangular banded matrix norm | matrix_norms.odin |
| lantp | c,d,s,z | Utility | Triangular packed matrix norm | matrix_norms.odin |
| lantr | c,d,s,z | Utility | Triangular matrix norm | matrix_norms.odin |
| lapmr | c,d,s,z | Utility | Rearrange rows of matrix | matrix_utilities.odin |
| lapmt | c,d,s,z | Utility | Rearrange columns of matrix | matrix_utilities.odin |
| larcm | c,z | Utility | Real matrix times complex matrix | auxiliary.odin |
| larf | c,d,s,z | Utility | Apply elementary Householder reflector | householder_reflectors.odin |
| larfb | c,d,s,z | Utility | Apply block Householder reflector | householder_reflectors.odin |
| larfg | c,d,s,z | Utility | Generate elementary Householder reflector | householder_reflectors.odin |
| larft | c,d,s,z | Utility | Form triangular factor of block Householder | householder_reflectors.odin |
| larfx | c,d,s,z | Utility | Apply elementary Householder reflector (auxiliary) | householder_reflectors.odin |
| larnv | c,d,s,z | Utility | Generate random vector | test_matrix_generation.odin |
| laror | c,d,s,z | Utility | Generate random orthogonal matrix | test_matrix_generation.odin |
| larot | c,d,s,z | Utility | Apply plane rotation to vectors | givens_rotations.odin |
| lartgp | d,s | Utility | Generate plane rotation with positive cosine | givens_rotations.odin |
| lartgs | d,s | Utility | Generate plane rotation with positive sine | givens_rotations.odin |
| lascl | c,d,s,z | Utility | Scale matrix by scalar | matrix_utilities.odin |
| laset | c,d,s,z | Utility | Initialize matrix to given values | matrix_utilities.odin |
| lasr | c,d,s,z | Utility | Apply sequence of plane rotations | givens_rotations.odin |
| lasrt | d,s | Utility | Sort real vector | auxiliary.odin |
| lassq | c,d,s,z | Utility | Update sum of squares for vector norm | matrix_norms.odin |
| latms | c,d,s,z | Utility | Generate test matrix with specified properties | test_matrix_generation.odin |
| latmt | c,d,s,z | Utility | Generate triangular test matrix | test_matrix_generation.odin |
| opgtr | d,s | Packed | Generate orthogonal matrix from Householder reflectors | packed_symmetric.odin |
| opmtr | d,s | Packed | Multiply by orthogonal matrix from Householder | packed_symmetric.odin |
| orbdb | d,s | Bidiagonal | Bidiagonal block diagonalization | bidiagonal.odin |
| orcsd | d,s | Orthogonal | Cosine-sine decomposition (CS) | orthogonal_transformations.odin |
| orgbr | d,s | Orthogonal | Generate orthogonal matrix from bidiagonal reduction | orthogonal_transformations.odin |
| orghr | d,s | Orthogonal | Generate orthogonal matrix from Hessenberg reduction | orthogonal_transformations.odin |
| orglq | d,s | Orthogonal | Generate Q from LQ factorization | orthogonal_transformations.odin |
| orgql | d,s | Orthogonal | Generate Q from QL factorization | orthogonal_transformations.odin |
| orgqr | d,s | Orthogonal | Generate Q from QR factorization | orthogonal_transformations.odin |
| orgrq | d,s | Orthogonal | Generate Q from RQ factorization | orthogonal_transformations.odin |
| orgtr | d,s | Orthogonal | Generate orthogonal matrix from tridiagonal reduction | orthogonal_transformations.odin |
| ormbr | d,s | Orthogonal | Multiply by orthogonal matrix from bidiagonal reduction | orthogonal_transformations.odin |
| ormhr | d,s | Orthogonal | Multiply by orthogonal matrix from Hessenberg reduction | orthogonal_transformations.odin |
| ormlq | d,s | Orthogonal | Multiply by Q from LQ factorization | orthogonal_transformations.odin |
| ormql | d,s | Orthogonal | Multiply by Q from QL factorization | orthogonal_transformations.odin |
| ormqr | d,s | Orthogonal | Multiply by Q from QR factorization | orthogonal_transformations.odin |
| ormrq | d,s | Orthogonal | Multiply by Q from RQ factorization | orthogonal_transformations.odin |
| ormrz | d,s | Orthogonal | Multiply by orthogonal matrix from RZ factorization | orthogonal_transformations.odin |
| ormtr | d,s | Orthogonal | Multiply by orthogonal matrix from tridiagonal reduction | orthogonal_transformations.odin |
| pbcon | c,d,s,z | Banded | Condition number of banded positive definite matrix | banded_positive_definite.odin |
| pbequ | c,d,s,z | Banded | Equilibration for banded positive definite matrix | banded_positive_definite.odin |
| pbrfs | c,d,s,z | Banded | Iterative refinement for banded positive definite system | banded_positive_definite.odin |
| pbstf | c,d,s,z | Banded | Cholesky factorization of split Cholesky form | banded_positive_definite.odin |
| pbsv | c,d,s,z | Banded | Positive definite banded linear system solver | banded_positive_definite.odin |
| pbsvx | c,d,s,z | Banded | Expert solver for banded positive definite system | banded_positive_definite.odin |
| pbtrf | c,d,s,z | Banded | Cholesky factorization of banded matrix | banded_positive_definite.odin |
| pbtrs | c,d,s,z | Banded | Solve banded positive definite system using Cholesky | banded_positive_definite.odin |
| pftrf | c,d,s,z | RFP | Cholesky factorization in RFP format | rfp.odin |
| pftri | c,d,s,z | RFP | Invert triangular matrix in RFP format | rfp.odin |
| pftrs | c,d,s,z | RFP | Solve system using RFP Cholesky factorization | rfp.odin |
| pocon | c,d,s,z | Dense | Condition number of positive definite matrix | dense_positive_definite.odin |
| poequ | c,d,s,z | Dense | Equilibration for positive definite matrix | dense_positive_definite.odin |
| poequb | c,d,s,z | Dense | Improved equilibration for positive definite matrix | dense_positive_definite.odin |
| porfs | c,d,s,z | Dense | Iterative refinement for positive definite system | dense_positive_definite.odin |
| porfsx | c,d,s,z | Dense | Expert iterative refinement for positive definite system | dense_positive_definite.odin |
| posv | c,d,s,z | Dense | Positive definite linear system solver | dense_positive_definite.odin |
| posvx | c,d,s,z | Dense | Expert solver for positive definite system | dense_positive_definite.odin |
| posvxx | c,d,s,z | Dense | Extra expert solver for positive definite system | dense_positive_definite.odin |
| ppcon | c,d,s,z | Packed | Condition number of packed positive definite matrix | packed_positive_definite.odin |
| ppequ | c,d,s,z | Packed | Equilibration for packed positive definite matrix | packed_positive_definite.odin |
| pprfs | c,d,s,z | Packed | Iterative refinement for packed positive definite system | packed_positive_definite.odin |
| ppsv | c,d,s,z | Packed | Positive definite packed linear system solver | packed_positive_definite.odin |
| ppsvx | c,d,s,z | Packed | Expert solver for packed positive definite system | packed_positive_definite.odin |
| pptrf | c,d,s,z | Packed | Cholesky factorization of packed matrix | packed_positive_definite.odin |
| pptri | c,d,s,z | Packed | Invert packed positive definite matrix using Cholesky | packed_positive_definite.odin |
| pptrs | c,d,s,z | Packed | Solve packed positive definite system using Cholesky | packed_positive_definite.odin |
| pstrf | c,d,s,z | Dense | Cholesky factorization with complete pivoting | dense_positive_definite.odin |
| ptcon | c,d,s,z | Tridiagonal | Condition number of positive definite tridiagonal matrix | tridiagonal_positive_definite.odin |
| pteqr | c,d,s,z | Tridiagonal | Eigenvalues/vectors of positive definite tridiagonal matrix | tridiagonal_eigenvalues.odin |
| ptrfs | c,d,s,z | Tridiagonal | Iterative refinement for positive definite tridiagonal system | tridiagonal_positive_definite.odin |
| ptsv | c,d,s,z | Tridiagonal | Positive definite tridiagonal linear system solver | tridiagonal_positive_definite.odin |
| ptsvx | c,d,s,z | Tridiagonal | Expert solver for positive definite tridiagonal system | tridiagonal_positive_definite.odin |
| pttrf | c,d,s,z | Tridiagonal | LDL factorization of tridiagonal matrix | tridiagonal_positive_definite.odin |
| pttrs | c,d,s,z | Tridiagonal | Solve positive definite tridiagonal system using LDL | tridiagonal_positive_definite.odin |
| sbev | d,s | Banded | Eigenvalues/vectors of symmetric banded matrix | banded_eigenvalues.odin |
| sbevd | d,s | Banded | Symmetric banded eigenvalues (divide-and-conquer) | banded_eigenvalues.odin |
| sbevx | d,s | Banded | Selected eigenvalues of symmetric banded matrix | banded_eigenvalues.odin |
| sbgst | d,s | Banded | Reduce generalized symmetric banded to standard form | banded.odin |
| sbgv | d,s | Banded | Generalized symmetric banded eigenvalue problem | banded_eigenvalues.odin |
| sbgvd | d,s | Banded | Generalized symmetric banded eigenvalues (divide-and-conquer) | banded_eigenvalues.odin |
| sbgvx | d,s | Banded | Selected generalized symmetric banded eigenvalues | banded_eigenvalues.odin |
| sbtrd | d,s | Banded | Reduce symmetric banded to tridiagonal form | banded_decomposition.odin |
| sfrk | d,s | RFP | Symmetric rank-k update in RFP format | rfp.odin |
| sgesv | d | Dense | Solve general linear system AX=B (single precision) | dense_linear.odin |
| spcon | c,d,s,z | Packed | Condition number of packed symmetric matrix | packed_symmetric_linear.odin |
| spev | d,s | Packed | Eigenvalues/vectors of packed symmetric matrix | packed_symmetric_eigenvalues.odin |
| spevd | d,s | Packed | Packed symmetric eigenvalues (divide-and-conquer) | packed_symmetric_eigenvalues.odin |
| spevx | d,s | Packed | Selected eigenvalues of packed symmetric matrix | packed_symmetric_eigenvalues.odin |
| spgst | d,s | Packed | Reduce generalized packed symmetric to standard form | packed_symmetric.odin |
| spgv | d,s | Packed | Generalized packed symmetric eigenvalue problem | packed_symmetric_eigenvalues.odin |
| spgvd | d,s | Packed | Generalized packed symmetric eigenvalues (divide-and-conquer) | packed_symmetric_eigenvalues.odin |
| spgvx | d,s | Packed | Selected generalized packed symmetric eigenvalues | packed_symmetric_eigenvalues.odin |
| sposv | d | Packed | Solve packed positive definite system (single precision) | packed_positive_definite.odin |
| sprfs | c,d,s,z | Packed | Iterative refinement for packed symmetric system | packed_symmetric_linear.odin |
| spsv | c,d,s,z | Packed | Symmetric indefinite packed linear system solver | packed_symmetric_linear.odin |
| spsvx | c,d,s,z | Packed | Expert solver for packed symmetric system | packed_symmetric_linear.odin |
| sptrd | d,s | Packed | Reduce packed symmetric to tridiagonal form | packed_symmetric.odin |
| sptrf | c,d,s,z | Packed | Bunch-Kaufman factorization of packed symmetric matrix | packed_symmetric.odin |
| sptri | c,d,s,z | Packed | Invert packed symmetric matrix using factorization | packed_symmetric_linear.odin |
| sptrs | c,d,s,z | Packed | Solve packed symmetric system using factorization | packed_symmetric_linear.odin |
| stebz | d,s | Tridiagonal | Selected eigenvalues of symmetric tridiagonal matrix | tridiagonal_eigenvalues.odin |
| stedc | c,d,s,z | Tridiagonal | Eigenvalues/vectors of symmetric tridiagonal matrix (divide-and-conquer) | tridiagonal_eigenvalues.odin |
| stegr | c,d,s,z | Tridiagonal | Eigenvalues/vectors of tridiagonal matrix (MRRR) | tridiagonal_eigenvalues.odin |
| stein | c,d,s,z | Tridiagonal | Eigenvectors by inverse iteration | tridiagonal_eigenvalues.odin |
| stemr | c,d,s,z | Tridiagonal | Eigenvalues/vectors by MRRR algorithm | tridiagonal_eigenvalues.odin |
| steqr | c,d,s,z | Tridiagonal | Eigenvalues/vectors of symmetric tridiagonal matrix | tridiagonal_eigenvalues.odin |
| sterf | d,s | Tridiagonal | Eigenvalues of symmetric tridiagonal matrix | tridiagonal_eigenvalues.odin |
| stev | d,s | Tridiagonal | Eigenvalues/vectors of symmetric tridiagonal matrix | tridiagonal_eigenvalues.odin |
| stevd | d,s | Tridiagonal | Symmetric tridiagonal eigenvalues (divide-and-conquer) | tridiagonal_eigenvalues.odin |
| stevr | d,s | Tridiagonal | Symmetric tridiagonal eigenvalues (RRR algorithm) | tridiagonal_eigenvalues.odin |
| stevx | d,s | Tridiagonal | Selected eigenvalues of symmetric tridiagonal matrix | tridiagonal_eigenvalues.odin |
| sycon | c,d,s,z | Dense | Condition number of symmetric matrix | dense_symmetric_linear.odin |
| syconv | c,d,s,z | Dense | Convert between symmetric matrix storage formats | dense_symmetric.odin |
| syequb | c,d,s,z | Dense | Improved equilibration for symmetric matrix | dense_symmetric_linear.odin |
| syev | d,s | Dense | Symmetric eigenvalue problem | dense_symmetric_eigenvalues.odin |
| syevd | d,s | Dense | Symmetric eigenvalue problem (divide-and-conquer) | dense_symmetric_eigenvalues.odin |
| syevr | d,s | Dense | Symmetric eigenvalue problem (RRR) | dense_symmetric_eigenvalues.odin |
| syevx | d,s | Dense | Symmetric eigenvalue problem (selected) | dense_symmetric_eigenvalues.odin |
| sygst | d,s | Dense | Reduce generalized symmetric eigenvalue problem to standard form | dense_symmetric_eigenvalues.odin |
| sygv | d,s | Dense | Generalized symmetric eigenvalue problem | dense_symmetric_eigenvalues.odin |
| sygvd | d,s | Dense | Generalized symmetric eigenvalue problem (divide-and-conquer) | dense_symmetric_eigenvalues.odin |
| sygvx | d,s | Dense | Generalized symmetric eigenvalue problem (selected) | dense_symmetric_eigenvalues.odin |
| syrfs | c,d,s,z | Dense | Iterative refinement for symmetric system | dense_symmetric_linear.odin |
| syrfsx | c,d,s,z | Dense | Expert iterative refinement for symmetric system | dense_symmetric_linear.odin |
| sysv | c,d,s,z | Dense | Symmetric indefinite linear system solver | dense_symmetric_linear.odin |
| sysvx | c,d,s,z | Dense | Expert solver for symmetric indefinite system | dense_symmetric_linear.odin |
| sysvxx | c,d,s,z | Dense | Extra expert solver for symmetric system | dense_symmetric_linear.odin |
| syswapr | c,d,s,z | Dense | Apply row/column swaps to symmetric matrix | dense_symmetric_linear.odin |
| sytrd | d,s | Dense | Reduce symmetric matrix to tridiagonal form | dense_symmetric_factorization.odin |
| sytrf | c,d,s,z | Dense | Bunch-Kaufman factorization of symmetric matrix | dense_symmetric_factorization.odin |
| sytri | c,d,s,z | Dense | Invert symmetric matrix using factorization | dense_symmetric_linear.odin |
| sytrs | c,d,s,z | Dense | Solve symmetric system using factorization | dense_symmetric_linear.odin |
| tbcon | c,d,s,z | Banded | Condition number of triangular banded matrix | triangular_linear.odin |
| tbrfs | c,d,s,z | Banded | Error bounds for triangular banded system solution | triangular_linear.odin |
| tbtrs | c,d,s,z | Banded | Solve triangular banded system | triangular_linear.odin |
| tfsm | c,d,s,z | RFP | Triangular solve with multiple right-hand sides (RFP) | rfp.odin |
| tftri | c,d,s,z | RFP | Invert triangular matrix in RFP format | rfp.odin |
| tfttp | c,d,s,z | RFP | Convert triangular matrix from RFP to packed format | rfp_conversion.odin |
| tfttr | c,d,s,z | RFP | Convert triangular matrix from RFP to standard format | rfp_conversion.odin |
| tgevc | c,d,s,z | Dense | Eigenvectors of generalized eigenvalue problem | dense_eigenvalues.odin |
| tgexc | c,d,s,z | Dense | Reorder generalized Schur factorization | dense_eigenvalues.odin |
| tgsen | c,d,s,z | Dense | Reorder generalized Schur form with condition numbers | dense_eigenvalues.odin |
| tgsja | c,d,s,z | Dense | Generalized SVD using Jacobi-type procedure | dense_svd.odin |
| tgsna | c,d,s,z | Dense | Condition numbers for generalized eigenvalue problem | dense_eigenvalues.odin |
| tgsyl | c,d,s,z | Dense | Solve generalized Sylvester equation | dense_linear.odin |
| tpcon | c,d,s,z | Packed | Condition number of packed triangular matrix | packed_triangular.odin |
| tplqt | c,d,s,z | Triangular | LQ factorization of triangular-pentagonal matrix | orthogonal_transformations.odin |
| tpmlqt | c,d,s,z | Triangular | Multiply by Q from triangular-pentagonal LQ | orthogonal_transformations.odin |
| tpmqrt | c,d,s,z | Triangular | Multiply by Q from triangular-pentagonal QR | orthogonal_transformations.odin |
| tpqrt | c,d,s,z | Triangular | QR factorization of triangular-pentagonal matrix | orthogonal_transformations.odin |
| tprfb | c,d,s,z | Packed | Apply block reflector to triangular-pentagonal matrix | orthogonal_transformations.odin |
| tprfs | c,d,s,z | Packed | Error bounds for packed triangular system solution | packed_triangular.odin |
| tptri | c,d,s,z | Packed | Invert packed triangular matrix | packed_triangular.odin |
| tptrs | c,d,s,z | Packed | Solve packed triangular system | packed_triangular.odin |
| tpttf | c,d,s,z | Packed | Convert triangular matrix from packed to RFP format | packed_triangular_conversion.odin |
| tpttr | c,d,s,z | Packed | Convert triangular matrix from packed to standard format | packed_triangular_conversion.odin |
| trcon | c,d,s,z | Triangular | Condition number of triangular matrix | triangular_linear.odin |
| trevc | c,d,s,z | Triangular | Eigenvectors of triangular matrix | dense_eigenvalues.odin |
| trexc | c,d,s,z | Triangular | Reorder Schur factorization | dense_eigenvalues.odin |
| trrfs | c,d,s,z | Triangular | Error bounds for triangular system solution | triangular_linear.odin |
| trsen | c,d,s,z | Triangular | Reorder Schur form with condition numbers | dense_eigenvalues.odin |
| trsna | c,d,s,z | Triangular | Condition numbers for eigenvalues/eigenvectors | dense_eigenvalues.odin |
| trsyl | c,d,s,z | Triangular | Solve Sylvester matrix equation | triangular_operations.odin |
| trtrs | c,d,s,z | Triangular | Solve triangular system | triangular_linear.odin |
| trttf | c,d,s,z | Triangular | Convert triangular matrix from standard to RFP format | triangular_operations.odin |
| trttp | c,d,s,z | Triangular | Convert triangular matrix from standard to packed format | triangular_operations.odin |
| tzrzf | c,d,s,z | Dense | RZ factorization of trapezoidal matrix | dense_factorization.odin |
| unbdb | c,z | Bidiagonal | Bidiagonal block diagonalization (complex unitary) | bidiagonal.odin |
| uncsd | c,z | Orthogonal | Cosine-sine decomposition (complex unitary) | orthogonal_transformations.odin |
| ungbr | c,z | Orthogonal | Generate unitary matrix from bidiagonal reduction | orthogonal_transformations.odin |
| unghr | c,z | Orthogonal | Generate unitary matrix from Hessenberg reduction | orthogonal_transformations.odin |
| unglq | c,z | Orthogonal | Generate Q from LQ factorization (complex) | orthogonal_transformations.odin |
| ungql | c,z | Orthogonal | Generate Q from QL factorization (complex) | orthogonal_transformations.odin |
| ungqr | c,z | Orthogonal | Generate Q from QR factorization (complex) | orthogonal_transformations.odin |
| ungrq | c,z | Orthogonal | Generate Q from RQ factorization (complex) | orthogonal_transformations.odin |
| ungtr | c,z | Orthogonal | Generate unitary matrix from tridiagonal reduction | orthogonal_transformations.odin |
| unmbr | c,z | Orthogonal | Multiply by unitary matrix from bidiagonal reduction | orthogonal_transformations.odin |
| unmhr | c,z | Orthogonal | Multiply by unitary matrix from Hessenberg reduction | orthogonal_transformations.odin |
| unmlq | c,z | Orthogonal | Multiply by Q from LQ factorization (complex) | orthogonal_transformations.odin |
| unmql | c,z | Orthogonal | Multiply by Q from QL factorization (complex) | orthogonal_transformations.odin |
| unmqr | c,z | Orthogonal | Multiply by Q from QR factorization (complex) | orthogonal_transformations.odin |
| unmrq | c,z | Orthogonal | Multiply by Q from RQ factorization (complex) | orthogonal_transformations.odin |
| unmrz | c,z | Orthogonal | Multiply by unitary matrix from RZ factorization | orthogonal_transformations.odin |
| unmtr | c,z | Orthogonal | Multiply by unitary matrix from tridiagonal reduction | orthogonal_transformations.odin |
| upgtr | c,z | Packed | Generate unitary matrix from Householder reflectors | packed_hermitian.odin |
| upmtr | c,z | Packed | Multiply by unitary matrix from Householder | packed_hermitian.odin |
