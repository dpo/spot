function result = ctranspose(A)
%'  Complex conjugate tranpose.
%   A' is the complex conjugate transpose of A.
%
%   CTRANSPOSE(A) is called for the syntax A' when A is a Sparco operator.

%   Copyright 2009, Ewout van den Berg and Michael P. Friedlander
%   http://www.cs.ubc.ca/labs/scl/sparco
%   $Id: ctranspose.m 41 2009-06-15 18:45:23Z ewout78 $

result = opCTranspose(A);