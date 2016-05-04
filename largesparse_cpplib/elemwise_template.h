
//! LVec is meant to be a very lightweight object for allocation on the stack as a local variable of a function.
//! It is to represent a view of a vector using
//! a data pointer declared as __restrict__. Thus for all the lifetime of this local object,
//! that data should not be accessed by anybody else. 
//! The intended usage for this class is to allow the compiler to efficiently optimize code that loops over access to elements.
//! such as e.g. the elementwise call.

template<class T>
class LVec
{
private:
 
  T* __restrict__ data_;
  int length_;
  int step_;


public:

  inline int length() const { return length_; }
  inline const T* data() const { return data_; }

  inline LVec(const LVec& v)
    :data_(v.data_), length_(v.length_), step_(v.step_)
  {}

  inline LVec(const T* data, int length, int step)
    :data_(data), length_(length), step_(step)
  {}

  inline LVec(const BVec<T>& v)
    :data_(v.data()), length_(v.length()), step_(v.step())
  {}

  //! Constructor from BVec also checking length matches the parameter
  inline LVec(const BVec<T>& v, int length)
    :data_(v.data()), length_(v.length()), step_(v.step())
  { assert(v.length()==length); }

  inline LVec(const T& val, int length)
    :data_(&val), length_(length), step_(0)
  {}  

  //! Initialize from scalar 
  inline LVec(const T& val, int length)
    :data_(&val), length_(length), step_(0)
  {}  

  inline T get(int i) const
  { return data_[step_*i]; }

  inline void put(int i, cons T& val) 
  { data_[step_*i] = val; }

};  

template<class T>
class typetraits
{
public:
  typedef T elemtype;
};

template<class T>
class typetraits< class BMat<T> >
{
public:
  typedef T elemtype;
};

template<class T>
class typetraits< class BVec<T> >
{
public:
  typedef T elemtype;
};


template<class T>
inline void check_pointers_are_different(const T* p1, const T* p2, const T* p3)
{
  assert (p2!=p1 );
}


template<class T>
inline void check_pointers_are_different(const T* p1, const T* p2, const T* p3)
{
  assert (p2!=p1  
          && p3!=p2 && p3!=p1
          );
}

template<class T>
inline void check_pointers_are_different(const T* p1, const T* p2, const T* p3, const T* p4)
{
  assert (p2!=p1  
          && p3!=p2 && p3!=p1
          && p4!=p3 && p4!=p2 && p4!=p1
          );
}

template<class T>
inline void check_pointers_are_different(const T* p1, const T* p2, const T* p3, const T* p4, const T*p5)
{
  assert (p2!=p1  
          && p3!=p2 && p3!=p1
          && p4!=p3 && p4!=p2 && p4!=p1
          && p5!=p4 && p5!=p3 && p5!=p2 && p5!=p1
          );
}

template<class T>
inline void check_pointers_are_different(const T* p1, const T* p2, const T* p3, const T* p4, const T*p5, const T*p6)
{
  assert (p2!=p1  
          && p3!=p2 && p3!=p1
          && p4!=p3 && p4!=p2 && p4!=p1
          && p5!=p4 && p5!=p3 && p5!=p2 && p5!=p1
          && p6!=p5 && p6!=p4 && p6!=p3 && p6!=p2 && p6!=p1
          );
}


//! Does elemwise: dest(i,j) = f( v1(i,j), v2(i,j), v3(i,j) )
template<class F, class V1, class V2, class V3, class Vdest>
inline void vector_elemwise_dest(const F& f, const V1& v1, const V2& v2, const V3& v3, Vdest& dest)
{
  LVec< typetraits<Vdest>::datatype > ldest(dest);
  int length = ldest.length();
  LVec< typetraits<V1>::datatype > lv1(v1, length); // check length or replicate it if it's a scalar
  LVec< typetraits<V2>::datatype > lv2(v2, length); // check length or replicate it if it's a scalar
  LVec< typetraits<V3>::datatype > lv3(v3, length); // check length or replicate it if it's a scalar

  // Verify no silly aliasing
  check_pointers_are_differnet(dest.data(), lv1.data(), lv2.data(), lv3.data());

  for(int i=0; i<length; i++)
    dest.put(i, f( lv1.get(i),
                   lv2.get(i),
                   lv3.get(i) ) );
}


//! A RowBVec is nothing but a BVec. But in operations  
//! where it makes sense to know whether we are dealing with a column vector (the default semantic of BVec)
//! or a row-vector, this class can be used to specify this is to be considered a row vector. 
template<class T>
class RowBVec: public BVec<T>
{
public:
  inline RowBVec(const BVec& v)
    :BVec(v)
  {}
};

template<class T> 
inline RowBVec<T> rowvec(const BVec<T>& v)
{ return RowBVec<T>(v); }

//! LMat is meant to be a lightweight object for allocation on the stack as a local variable of a function.
//! It is to represent a view of a matrix using
//! a data pointer declared as __restrict__. Thus for all the lifetime of this local object,
//! that data should not be accessed by anybody else. 
//! The intended usage for this class is to allow the compiler to efficiently optimize code that loops over access to elements.
//! such as e.g. the elementwise call.
//! Note that this matrix class allows for strides in both rows and columns
//! It also allows for either one or both of these strides to be 0, with the meaning that the data is replicated along that dimension
//! This way it can represent a replicated row, column, or even replicated scalar.

template<class T>
class LMat
{
private:
 
  T* __restrict__ data_;
  int nrows_;
  int ncols_;
  int row_step_; // step to next row
  int col_step_; // step to next column

public:

  inline const T* data() const { return data_; }

  inline int nrows() const { return nrows_; }
  inline int ncols() const { return ncols_; }
  inline int row_step() const { return row_step_; }
  inline int col_step() const { return col_step_; }

  //! Constructorfrom bare pointer
  inline LMat(const T* data, int nrows, int ncols, int row_step, int col_step)
    :data_(data), nrows_(nrows), ncols_(ncols), 
     row_step_(row_step), col_step_(col_step)
  {}

  inline LMat(const LMat<T>& m)
    :data_(m.data_), nrows_(m.nrows_), ncols_(m.ncols_), row_step_(m.row_step_), col_step_(m.col_step_)
  {}

  //! Constructor from BMat
  inline LMat(const BMat<T>& m)
    :data_(m.data()), nrows_(m.nrows()), ncols_(m.ncols()), row_step_(1), col_step_(m.stride())
  {}

  //! Constructor from BMat, checking dimensions
  inline LMat(const BMat<T>& m, int nrows, int ncols)
    :data_(m.data()), nrows_(m.nrows()), ncols_(m.ncols()), row_step_(1), col_step_(m.stride())
  {
    assert(nrows==m.nrows() && ncols==m.ncols());
  }


  //! Constructor from BVec
  inline LMat(const BVec<T>& v)
    :data_(v.data()), nrows_(v.length()), ncols_(1), row_step_(v.step()), col_step_(0)
  {}

  //! Constructor from BVec: replicated column vector
  inline LMat(const BVec<T>& v, int nrows, int ncols)
    :data_(v.data()), nrows_(nrows), ncols_(ncols), row_step_(v.step()), col_step_(0)
  {
    assert(nrows==v.length());
  }

  //! Constructor from RowBVec
  inline LMat(const RowBVec<T>& v)
    :data_(v.data()), nrows_(1), ncols_(v.length()), row_step_(0), col_step_(v.step())
  {}

  //! Constructor from RowBVec: replicated row vector
  inline LMat(const RowBVec<T>& v, int nrows, int ncols)
    :data_(v.data()), nrows_(nrows), ncols_(ncols), row_step_(0), col_step_(v.step())
  {
    assert(ncols==v.length());
  }

  //! constructor from scalar
  inline epxlicit LMat(const T& s)
    :data_(&s), nrows_(1), ncols_(1), row_step(0), col_step(0)
  {}

  //! constructor from scalar: replicated scalar
  inline LMat(const Scalar& s, int nrows, int ncols)
    :data_(&s), nrows_(nrows), ncols_(ncols), row_step(0), col_step(0)
  {}
  
  //! Note: by design, get and put replicate rows and/or columns when row_sep_==0 or col_sep_==0
  inline T get(int i, int j) const
  { return data_[i*row_step_ + j*col_step_]; }

  inline void put(int i, int j, const T& val) 
  { data_[i*row_step_ + j*col_step_] = val; }

};  


//! Does elemwise: m1(i,j) = f( m1(i,j), m2(i,j), m3(i,j) )
template<class F, class M1, class M2, class M3, class Mdest>
inline void matrix_elemwise_inplace(const F& f, const M1& m1, const M2& m2, const M3& m3)
{
  LMat< typetraits<M1>::datatype > lm1(m1); 
  int nrows = lm1.nrows();
  int ncols = lm1.ncols();
  LMat< typetraits<M2>::datatype > lm2(m2, nrows, ncols); // check compatibility and extend it to nrows x ncols if necessary
  LMat< typetraits<M3>::datatype > lm3(m3, nrows, ncols); // check compatibility and extend it to nrows x ncols if necessary

  // Verify no silly aliasing
  check_pointers_are_differnet(dest.data(), lm1.data(), lm2.data(), lm3.data());


  // Heuristically estimate the better looping order
  if (ldest.row_step() < ldest.col_step() ) // use column-major visiting order
    {
      for (int j=0; j<ncols; j++)
        for(int i=0; i<nrows; i++)
          m1.put(i, j, f( lm1.get(i,j),
                          lm2.get(i,j),
                          lm3.get(i,j) ) );
    }
  else  // use row-major visiting order
    {
      for(int i=0; i<nrows; i++)
        for (int j=0; j<ncols; j++)
          m1.put(i, j, f( lm1.get(i,j),
                          lm2.get(i,j),
                          lm3.get(i,j) ) );
    }

}


//! Does elemwise: dest(i,j) = f( m1(i,j), m2(i,j), m3(i,j) )
template<class F, class M1, class M2, class M3, class Mdest>
inline void matrix_elemwise_dest(const F& f, const M1& m1, const M2& m2, const M3& m3, Mdest& dest)
{
  LMat< typetraits<Mdest>::datatype > ldest(dest);
  int nrows = ldest.nrows();
  int ncols = ldest.ncols();
  LMat< typetraits<M1>::datatype > lm1(m1, nrows, ncols); // check compatibility and extend it to nrows x ncols if necessary
  LMat< typetraits<M2>::datatype > lm2(m2, nrows, ncols); // check compatibility and extend it to nrows x ncols if necessary
  LMat< typetraits<M3>::datatype > lm3(m3, nrows, ncols); // check compatibility and extend it to nrows x ncols if necessary

  // Verify no silly aliasing
  check_pointers_are_different(dest.data(), lm1.data(), lm2.data(), lm3.data());


  // Heuristically estimate the better looping order
  if (ldest.row_step() < ldest.col_step() ) // use column-major visiting order
    {
      for (int j=0; j<ncols; j++)
        for(int i=0; i<nrows; i++)
          dest.put(i, j, f( lm1.get(i,j),
                            lm2.get(i,j),
                            lm3.get(i,j) ) );
    }
  else  // use row-major visiting order
    {
      for(int i=0; i<nrows; i++)
        for (int j=0; j<ncols; j++)
          dest.put(i, j, f( lm1.get(i,j),
                            lm2.get(i,j),
                            lm3.get(i,j) ) );
    }

}



Ex:

// Simple functor

template<class T1, class T2, class T3>
class a_plus_b_c
{
  T operator()(const T1& a, const T2& b, const T3& c)
  { return a + b*c; }
};

template<class T1, class T2, class T3>
viod rank1update(BMat<T1>& m, const BVec<T2>& u, const BVec<T3>& v)
{
  matrix_elemwise_inplace(a_plus_b_c(), m, u, rowvec(v) );
}


