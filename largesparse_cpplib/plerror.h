// -*- C++ -*-

// PLearn (A C++ Machine Learning Library)
// Copyright (C) 1998 Pascal Vincent
// Copyright (C) 1999-2002 Pascal Vincent, Yoshua Bengio and University of Montreal
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
//
// This file is part of the PLearn library. For more information on the PLearn
// library, go to the PLearn Web site at www.plearn.org




/* *******************************************************
 * $Id: plerror.h 9705 2008-11-20 20:12:05Z nouiz $
 * AUTHORS: Pascal Vincent & Yoshua Bengio
 * This file is part of the PLearn library.
 ******************************************************* */


/*! \file plerror.h */

#ifndef perror_INC
#define perror_INC

#include <stdlib.h>
#include <iostream>     // std::cout, std::ostream, std::hex
#include <sstream>      // std::stringbuf
#include <string>       // std::string

#include "plexceptions.h"

#ifndef __GNUC__
// Suppress __attribute__(()) GCC extension on other compilers.
#define __attribute__(x)
#endif


namespace PLearn {

using std::cerr;
using std::cout;
using std::endl;

// #define PLERROR(...); { cerr << "PLERROR at " << __FILE__ << ":" << __LINE__ << "  " <<  __VA_ARGS__ << endl; exit(1); } 
#define PLERROR(...) ( (cerr << "PLERROR at " << __FILE__ << ":" << __LINE__ << "  " <<  __VA_ARGS__ << endl), exit(1) )  


// Redefine the assert mechanism to throw an exception through PLERROR.
// The following macros are defined:
//
// 1) PLASSERT:     same syntax as standard assert(), but throws exception
//
// 2) PLASSERT_MSG: accepts a second argument (std::string) which indicates
//                  a cause for the assertion failure.  If one needs to
//                  perform complex formatting on that string (substitute
//                  variables, etc.), it is recommended to use the Boost
//                  'format' library.

// When debugging, do nothing (do static cast as in GCC)
#ifdef  NDEBUG

#  define PLASSERT(expr) static_cast<void>(0)
#  define PLASSERT_MSG(expr, message) static_cast<void>(0)

#else   // ! defined(NDEBUG)

#  define PLASSERT(expr)                                                    \
   static_cast<void>((expr) ? 0 :                                           \
                     (PLearn::pl_assert_fail(#expr, __FILE__, __LINE__,     \
                                             PL_ASSERT_FUNCTION, ""), 0))

#  define PLASSERT_MSG(expr, message)                                       \
   static_cast<void>((expr) ? 0 :                                           \
                     (PLearn::pl_assert_fail(#expr, __FILE__, __LINE__,     \
                                             PL_ASSERT_FUNCTION, (message)), 0))

#  define PL_ASSERT_DEFINED

#endif  // NDEBUG

// Similarly, define PLCHECK mechanism to perform some checks. These checks
// will be done even if NDEBUG is defined.

#  define PLCHECK(expr)                                                     \
   static_cast<void>((expr) ? 0 :                                           \
                     (PLearn::pl_check_fail(#expr, __FILE__, __LINE__,      \
                                             PL_ASSERT_FUNCTION, ""), 0))

#  define PLCHECK_MSG(expr, message)                                        \
   static_cast<void>((expr) ? 0 :                                           \
                     (PLearn::pl_check_fail(#expr, __FILE__, __LINE__,      \
                                            PL_ASSERT_FUNCTION, (message)), 0))

#  define PL_ASSERT_DEFINED


// Use the function prettification code present in GCC's assert.h include
#if defined __USE_GNU
#  define PL_ASSERT_FUNCTION    __PRETTY_FUNCTION__
#else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#    define PL_ASSERT_FUNCTION  __func__
#  else
#    define PL_ASSERT_FUNCTION  ((__const char *) 0)
#  endif
#endif


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
