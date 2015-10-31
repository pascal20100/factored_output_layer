// -*- C++ -*-

// blas_linalg.h
//
// Copyright (C) 2014 Pascal Vincent and Universite de Montreal
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
// 
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
// 
//  3. The name of the authors may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
// NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#ifndef DataBatchStream_INC
#define DataBatchStream_INC

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <string>

#include "plerror.h"
#include <PP.h>
#include <k_sparse.h>

namespace PLearn {

using std::string;

inline FILE* safe_fopen(string fname, string mode="rb")
{
  FILE* f = fopen(fname.c_str(),mode.c_str());
  return f;
}

inline long file_size(string fname)
{
  FILE* f = safe_fopen(fname);
  fseek(f, 0, SEEK_END);
  long fsize =  ftell(f);
  fclose(f);
  return fsize;
}

template<class T>
class KSparseDataBatchStream : public PPointable
{
protected:
    int n_examples_per_epoch_;
    int batch_size_;
    int nepochs_;
    int input_d_;
    int input_K_;
    int target_d_;
    int target_K_;
    
public:
    
    KSparseDataBatchStream(int n, int m, int nepochs,
                           int id, int iK, int td, int tK)
        :n_examples_per_epoch_(n), batch_size_(m), nepochs_(nepochs),
         input_d_(id), input_K_(iK), target_d_(td), target_K_(tK) 
    {}

    //! returns how many examples are in the dataset (or -1 if unspecified)
    inline int n_examples() const
    { return n_examples_per_epoch_; }

    //! returns how many examples are in each batch
    inline int batch_size() const
    { return batch_size_; }

    //! returns the dimensionality of the inputs
    inline int input_d() const 
    { return input_d_; }

    //! returns the K-sparsity degree of the inputs
    inline int input_K() const 
    { return input_K_; }

    //! returns the dimensionality of the targets
    inline int target_d() const 
    { return target_d_; }

    //! returns the K-sparsity degree of the targets
    inline int target_K() const
    { return target_K_; }

    //! Returns for how many epochs the stream will send minibatches
    //! all together it will send n_epochs()*n_examples()/batch_size() minibatches.
    inline int n_epochs() const
    { return nepochs_; }

    //! Will copy the next minibatch into the columns of input.values, input.indexes, target.values, and target_indexe
    //! This call performs some basic dimensionality checks and then calls get_batch_ which sould be redefined by subclasses to do the real work.
    void get_next_batch(const CKSparseMat<T>& input, const CKSparseMat<T>& target)
    {
        assert(input.nrows() == input_d());
        assert(target.nrows() == target_d());
        assert(input.values.nrows() == input_K() && input.indexes.nrows() == input_K() );
        assert(target.values.nrows() == target_K() && target.indexes.nrows() == target_K() );
        assert(input.values.ncols() == batch_size() && input.indexes.ncols() == batch_size() && 
               target.values.ncols() == batch_size() && target.indexes.ncols() == batch_size());
        get_next_batch_(input, target);
    }

  virtual ~KSparseDataBatchStream() 
  {}

protected:
    //! Redefine in subclasses to fill the columns of the given matrices with dataset examples
    virtual void get_next_batch_(const CKSparseMat<T>& input, const CKSparseMat<T>& target) = 0;    
};

//! A KSparseDataBatchStream from data matrices residing in memory.
template<class T>
class MemoryKSparseDataBatchStream: public KSparseDataBatchStream<T>
{
protected:
    CKSparseMat<T> inputdata;
    CKSparseMat<T> targetdata;
    int pos;
    
public:   
    //! Examples are in the columns of the matrices
    MemoryKSparseDataBatchStream(int m, const CKSparseMat<T>& inputs, const CKSparseMat<T>& targets):
        KSparseDataBatchStream<T>(inputs.ncols(), m, -1,
                                  inputs.nrows(), inputs.K(), targets.nrows(), targets.K() ),
        inputdata(inputs),
        targetdata(targets),
        pos(0)
    {
    }


protected:
    //! Redefine in subclasses to fill the columns of the given matrices with dataset examples
    virtual void get_next_batch_(const CKSparseMat<T>& input, const CKSparseMat<T>& target)
    {
        // int m = batch_size();
        int m = this->batch_size_;
        if (pos+m <= inputdata.values.ncols())
        {
            input.values << inputdata.values.subMatColumns(pos,m);
            input.indexes << inputdata.indexes.subMatColumns(pos,m);
            target.values << targetdata.values.subMatColumns(pos,m);
            target.indexes << targetdata.indexes.subMatColumns(pos,m);
            pos += m;
            if (pos == inputdata.values.ncols())
                pos = 0;
        }
        else // we're getting a chunk that's part of the end of the dataset and looping around to part of the beginning
        {
            // how many to copy from the end
            int endm = inputdata.values.ncols() - pos;
            input.values.subMatColumns(0,endm) << inputdata.values.subMatColumns(pos,endm);
            input.indexes.subMatColumns(0,endm) << inputdata.indexes.subMatColumns(pos,endm);
            target.values.subMatColumns(0,endm) << targetdata.values.subMatColumns(pos,endm);
            target.indexes.subMatColumns(0,endm) << targetdata.indexes.subMatColumns(pos,endm);
            
            // how many to copy from the beginning
            int begm = m-endm;
            input.values.subMatColumns(endm,begm) << inputdata.values.subMatColumns(0,begm);
            input.indexes.subMatColumns(endm,begm) << inputdata.indexes.subMatColumns(0,begm);
            target.values.subMatColumns(endm,begm) << targetdata.values.subMatColumns(0,begm);
            target.indexes.subMatColumns(endm,begm) << targetdata.indexes.subMatColumns(0,begm);
            
            pos = begm;
        }
    }
};




/*
//! A KSparseDataBatchStream from data matrices residing in files.
//! Calls to get_next_bath will fill batches with value matrices type BMat<T> and index matrices of type BMat<int>
//! But the data types stored in the files may actually be different and will correspond to Tval and Tindex (e.g. unsigned short for Tindex) 
template<class T, class Tval, class Tindex>
class FilesKSparseDataBatchStream: public KSparseDataBatchStream<T>
{
protected:

  string input_values_fname_;
  string input_indexes_fname_;
  string target_values_fname_;
  string target_indexes_fname_;

  long input_values_offset_;   // start position offset in bytes
  long input_indexes_offset_;  // start position offset in bytes
  long target_values_offset_;  // start position offset in bytes
  long target_indexes_offset_; // start position offset in bytes
  
  FILE* input_values_f_;
  FILE* input_indexes_f_;
  FILE* target_values_f_;
  FILE* target_indexes_f_;

  // BMatrices to be filled directly from reading form file
  // and whose element types match those stored in the file
  BMat<Tval> input_values_;
  BMat<Tindex> input_indexes_;
  BMat<Tval> target_values_;
  BMat<Tindex> target_indexes_;

  long pos_; // current example number
  long n_examples_; // number of examples (length of dataset)
  
public:    

  //! If n_examples <= 0 the number of examples will be determined automatically from the file sizes
  //! (in that case, files must contain nothing but the data after the optionally specified offset).
    FilesKSparseDataBatchStream(int m, int id, int iK, int td, int tK, 
                              string input_values_fname, string input_indexes_fname, 
                              string target_values_fname, string target_indexes_fname,
                              long input_values_offset=0, long input_indexes_offset=0, 
                              long target_values_offset=0, long target_indexes_offset=0, 
                              long n_examples = -1):
        KSparseDataBatchStream<T>(m, id, iK, td, tK),
        input_values_fname_(input_values_fname),
        input_indexes_fname_(input_indexes_fname),
        target_values_fname_(target_values_fname),
        target_indexes_fname_(target_indexes_fname),
        input_values_offset_(input_values_offset),
        input_indexes_offset_(input_indexes_offset),
        target_values_offset_(target_values_offset),
        target_indexes_offset_(target_indexes_offset),
        input_values_(iK,m),
        input_indexes_(iK,m),
        target_values_(tK,m),
        target_indexes_(tK,m),
        pos_(0)
    {
      if (n_examples<0) // let us determine the number of examples form the file sizes and check for consistency among them
        {
          n_examples = (file_size(input_indexes_fname_)-input_indexes_offset_)/( iK*sizeof(Tindex));
          assert( n_examples*iK*sizeof(Tval) + input_values_offset_ == file_size(input_values_fname_) );
          assert( n_examples*iK*sizeof(Tindex) + input_indexes_offset_ == file_size(input_indexes_fname_) );
          assert( n_examples*tK*sizeof(Tval) + target_values_offset_ == file_size(target_values_fname_) );
          assert( n_examples*tK*sizeof(Tindex) + target_indexes_offset_ == file_size(target_indexes_fname_) );
        }
      // if n>0 the user specified the number of examples. We suppose he knows what he is doing and
      // he may be pointing to a part inside a file that could contain other things.
      // So we cannot rely on file sizes matching n_examples. 

      n_examples_ = n_examples;

      input_values_f_ = safe_fopen(input_values_fname_);
      input_indexes_f_ = safe_fopen(input_indexes_fname_);
      target_values_f_ = safe_fopen(target_values_fname_);
      target_indexes_f_ = safe_fopen(target_indexes_fname_);      

      this->seek_to_beginning();
    }


protected:  

  void seek_to_beginnning() 
  {
    fseek(input_values_f_, input_values_offset_, SEEK_SET);
    fseek(input_indexes_f_, input_indexes_offset_, SEEK_SET);    
    fseek(target_values_f_, target_values_offset_, SEEK_SET);
    fseek(target_indexes_f_, target_indexes_offset_, SEEK_SET);    
  }

  void copy_next_batch_into_buffers() 
  {
    assert(input_values_.is_contiguous() && input_indexes_.is_contiguous() && target_values_.is_contiguous() && target_indexes_.is_contiguous());
    
    int m = this->batch_size();
    int iK = this->input_K();
    int tK = this->target_K();

    if (pos_+m <= n_examples_)
      {
        fread(input_values_.data(), sizeof(Tval), m*iK, input_values_f_);
        fread(input_indexes_.data(), sizeof(Tindex), m*iK, input_indexes_f_);
        fread(target_values_.data(), sizeof(Tval), m*tK, target_values_f_);
        fread(target_indexes_.data(), sizeof(Tindex), m*tK, target_indexes_f_);

        pos_ += m;
        if (pos_ == n_examples_)
          {
            pos_ = 0;
            this->seek_to_beginning();
          }
      }
    else // we're getting a chunk that's part of the end of the files and looping around to part of the beginning
      {
        // how many to copy till the end
        long endm = n_examples_ - pos_;
        fread(input_values_.data(), sizeof(Tval), endm*iK, input_values_f_);
        fread(input_indexes_.data(), sizeof(Tindex), endm*iK, input_indexes_f_);
        fread(target_values_.data(), sizeof(Tval), endm*tK, target_values_f_);
        fread(target_indexes_.data(), sizeof(Tindex), endm*tK, target_indexes_f_);
        
        // how many to copy from the beginning
        long begm = m-endm;
        this->seek_to_beginning();
        fread(input_values_.data() + input_values_.stride()*sizeof(Tval), sizeof(Tval), begm*iK, input_values_f_);
        fread(input_indexes_.data() + input_indexes_.stride()*sizeof(Tindex), sizeof(Tindex), begm*iK, input_indexes_f_);
        fread(target_values_.data() + target_values_.stride()*sizeof(Tval), sizeof(Tval), begm*tK, target_values_f_);
        fread(target_indexes_.data() + target_indexes_.stride()*sizeof(Tindex), sizeof(Tindex), begm*tK, target_indexes_f_);
        
        pos_ = begm;
      }
    }

  void get_next_batch_(const CKSparseMat<T>& input, const CKSparseMat<T>& target)
  {
    copy_next_batch_into_buffers();
    // These copies should effect type conversion if necessary
    input.values << input_values_;
    input.indexes << input_indexes_;
    target.values << target_values_;
    target.indexes << target_indexes_;
  }

public:
  ~FilesKSparseDataBatchStream() 
  {
  }
  
};
*/

//! A KSparseDataBatchStream read form a FILE* (like e.g. stdin).
//! This is actually meant for receiving batches from a process that continuously outpus data rather than reading the data from a file.
//! All integers sent are supposed to correspond to the int type of the platform for both sender and receiver (typically 32 bit signed integer)
//! The sender is supposed to first send a header of 9 ints:
//!   - header start is magic int 9876543210
//!   - n_examples_per_epoch
//!   - batch_size
//!   - nepochs   number of epochs that willl be streamed
//!   - input_d
//!   - input_K
//!   - target_d
//!   - target_K
//!   - size of value type in bytes
//! Note that it is the sender that thus defines the length of the batches. The user of this class must adapt to it 
//! (i.e. subsequently passing matrices with that number of columns to be filled when it calls get_next_batch(...) )
//!
//! This is followed by the sender sending an unending sequence of batches containing each batch_size examples. Each batch is sent as follows:
//!   - batch start header is magic int 123456789
//!   - batch_size*input_K input values of type T (the input_K values for the first example of the batch, followed by those of the second, etc...)
//!   - batch_size*input_K input indexes of type int
//!   - batch_size*target_K target values of type T (the target_K values for the first example of the batch, followed by those of the second, etc...)
//!   - batch_size*target_K target indexes of type int
template<class T>
class PipedKSparseDataBatchStream: public KSparseDataBatchStream<T>
{
protected:
    FILE* f_;
    
public:
    
    PipedKSparseDataBatchStream(FILE* f):
        KSparseDataBatchStream<T>(-1, -1, -1, -1, -1, -1, -1), 
        f_(f)
    {
        assert( sizeof(int)==32/8 );

        int header[9]; 
        fread(header, sizeof(int), 9, f_);
        assert( header[0] == 987654321 );
        this->n_examples_per_epoch_ = header[1];
        this->batch_size_ = header[2];
        this->nepochs_ = header[3];
        this->input_d_ = header[4];
        this->input_K_ = header[5];
        this->target_d_ = header[6];
        this->target_K_ = header[7];
        int value_size = header[8];
        
        if (value_size != sizeof(T) )
            PLERROR("ERROR: stream will send elements of size " << value_size << " bytes, while the class was instantiated with element types of size " << int(sizeof(T)) << " bytes (incompatible element type float or bouble?)\n");
    }


protected:

    void get_next_batch_(const CKSparseMat<T>& input, const CKSparseMat<T>& target)
    {
        assert(input.values.is_contiguous() && input.indexes.is_contiguous() && target.values.is_contiguous() && target.indexes.is_contiguous());

        int batch_header[2];
        fread( batch_header, sizeof(int), 2, f_);
        if ( batch_header[0] != 123456789 )
            PLERROR("ERROR: expected to read batch start header of value 1234567 bu instead read " << batch_header[0]); 
        if ( batch_header[1] != this->batch_size_ )
            PLERROR("Second int of batch header does not match batch_size received in initial stream header. Currently receiving batches of varying number of examples is not yet supported."); 

        int m = this->batch_size_;
        int iK = this->input_K_;
        int tK = this->target_K_;

        fread( input.values.data(), sizeof(T), iK * m, f_);
        fread( input.indexes.data(), sizeof(int), iK * m, f_);
        fread( target.values.data(), sizeof(T), tK * m, f_);
        fread( target.indexes.data(), sizeof(int), tK * m, f_);
    }
};


} // end of namespace PLearn


#endif

  /*
    Local Variables:
    mode:c++
    c-basic-offset:4
    c-file-style:"stroustrup"
    c-file-offsets:((innamespace . 0)(inline-open . 0))
    indent-tabs-mode:nil
    fill-column:79
    End:
  */
  // vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:encoding=utf-8:textwidth=79 :
