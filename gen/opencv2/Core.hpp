/** @brief Copies the lower or the upper half of a square matrix to its another half.

The function cv::completeSymm copies the lower or the upper half of a square matrix to
its another half. The matrix diagonal remains unchanged:
 - \f$\texttt{m}_{ij}=\texttt{m}_{ji}\f$ for \f$i > j\f$ if
    lowerToUpper=false
 - \f$\texttt{m}_{ij}=\texttt{m}_{ji}\f$ for \f$i < j\f$ if
    lowerToUpper=true

@param m input-output floating-point square matrix.
@param lowerToUpper operation flag; if true, the lower half is copied to
the upper half. Otherwise, the upper half is copied to the lower half.
@sa flip, transpose
*/
CV_EXPORTS_W void completeSymm(InputOutputArray m, bool lowerToUpper = false);

/** @brief Scales, calculates absolute values, and converts the result to 8-bit.

On each element of the input array, the function convertScaleAbs
performs three operations sequentially: scaling, taking an absolute
value, conversion to an unsigned 8-bit type:
\f[\texttt{dst} (I)= \texttt{saturate\_cast<uchar>} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\f]
In case of multi-channel arrays, the function processes each channel
independently. When the output is not 8-bit, the operation can be
emulated by calling the Mat::convertTo method (or by using matrix
expressions) and then by calculating an absolute value of the result.
For example:
@code{.cpp}
    Mat_<float> A(30,30);
    randu(A, Scalar(-100), Scalar(100));
    Mat_<float> B = A*5 + 3;
    B = abs(B);
    // Mat_<float> B = abs(A*5+3) will also do the job,
    // but it will allocate a temporary matrix
@endcode
@param src input array.
@param dst output array.
@param alpha optional scale factor.
@param beta optional delta added to the scaled values.
@sa  Mat::convertTo, cv::abs(const Mat&)
*/
CV_EXPORTS_W void convertScaleAbs(InputArray src, OutputArray dst,
                                  double alpha = 1, double beta = 0);


/** @example copyMakeBorder_demo.cpp
An example using copyMakeBorder function
 */
/** @brief Forms a border around an image.

The function copies the source image into the middle of the destination image. The areas to the
left, to the right, above and below the copied source image will be filled with extrapolated
pixels. This is not what filtering functions based on it do (they extrapolate pixels on-fly), but
what other more complex functions, including your own, may do to simplify image boundary handling.

The function supports the mode when src is already in the middle of dst . In this case, the
function does not copy src itself but simply constructs the border, for example:

@code{.cpp}
    // let border be the same in all directions
    int border=2;
    // constructs a larger image to fit both the image and the border
    Mat gray_buf(rgb.rows + border*2, rgb.cols + border*2, rgb.depth());
    // select the middle part of it w/o copying data
    Mat gray(gray_canvas, Rect(border, border, rgb.cols, rgb.rows));
    // convert image from RGB to grayscale
    cvtColor(rgb, gray, COLOR_RGB2GRAY);
    // form a border in-place
    copyMakeBorder(gray, gray_buf, border, border,
                   border, border, BORDER_REPLICATE);
    // now do some custom filtering ...
    ...
@endcode
@note When the source image is a part (ROI) of a bigger image, the function will try to use the
pixels outside of the ROI to form a border. To disable this feature and always do extrapolation, as
if src was not a ROI, use borderType | #BORDER_ISOLATED.

@param src Source image.
@param dst Destination image of the same type as src and the size Size(src.cols+left+right,
src.rows+top+bottom) .
@param top
@param bottom
@param left
@param right Parameter specifying how many pixels in each direction from the source image rectangle
to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs
to be built.
@param borderType Border type. See borderInterpolate for details.
@param value Border value if borderType==BORDER_CONSTANT .

@sa  borderInterpolate
*/
CV_EXPORTS_W void copyMakeBorder(InputArray src, OutputArray dst,
                                 int top, int bottom, int left, int right,
                                 int borderType, const Scalar& value = Scalar() );
/** @brief Performs a forward or inverse discrete Cosine transform of 1D or 2D array.

The function cv::dct performs a forward or inverse discrete Cosine transform (DCT) of a 1D or 2D
floating-point array:
-   Forward Cosine transform of a 1D vector of N elements:
    \f[Y = C^{(N)}  \cdot X\f]
    where
    \f[C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right )\f]
    and
    \f$\alpha_0=1\f$, \f$\alpha_j=2\f$ for *j \> 0*.
-   Inverse Cosine transform of a 1D vector of N elements:
    \f[X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y\f]
    (since \f$C^{(N)}\f$ is an orthogonal matrix, \f$C^{(N)} \cdot \left(C^{(N)}\right)^T = I\f$ )
-   Forward 2D Cosine transform of M x N matrix:
    \f[Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T\f]
-   Inverse 2D Cosine transform of M x N matrix:
    \f[X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)}\f]

The function chooses the mode of operation by looking at the flags and size of the input array:
-   If (flags & #DCT_INVERSE) == 0 , the function does a forward 1D or 2D transform. Otherwise, it
    is an inverse 1D or 2D transform.
-   If (flags & #DCT_ROWS) != 0 , the function performs a 1D transform of each row.
-   If the array is a single column or a single row, the function performs a 1D transform.
-   If none of the above is true, the function performs a 2D transform.

@note Currently dct supports even-size arrays (2, 4, 6 ...). For data analysis and approximation, you
can pad the array when necessary.
Also, the function performance depends very much, and not monotonically, on the array size (see
getOptimalDFTSize ). In the current implementation DCT of a vector of size N is calculated via DFT
of a vector of size N/2 . Thus, the optimal DCT size N1 \>= N can be calculated as:
@code
    size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }
    N1 = getOptimalDCTSize(N);
@endcode
@param src input floating-point array.
@param dst output array of the same size and type as src .
@param flags transformation flags as a combination of cv::DftFlags (DCT_*)
@sa dft , getOptimalDFTSize , idct
*/
CV_EXPORTS_W void dct(InputArray src, OutputArray dst, int flags = 0);
