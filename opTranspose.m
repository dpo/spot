%opFoG   Forms the produce of to operators.
%
%   opFoG(OP1,OP2) creates an operator that successively applies each
%   of the operators OP1, OP2 on a given input vector. In non-adjoint
%   mode this is done in reverse order.
%
%   The inputs must be either Spot operators or explicit Matlab matrices
%   (including scalars).
%
%   See also opDictionary, opStack, opSum.

%   Copyright 2009, Ewout van den Berg and Michael P. Friedlander
%   http://www.cs.ubc.ca/labs/scl/sparco
%   $Id: opFoG.m 39 2009-06-12 20:59:05Z ewout78 $

classdef opFoG < opSpot

   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % Constructor
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function op = opFoG(A,B)
          
          if nargin == 0
             error('At least one operator must be specified.')
          end
           
          % Make sure that there are at least two operators.
          if nargin == 1
             B = [];
          end
          
          % Input matrices are immediately cast as opMatrix's.
          if isa(A,'numeric'), A = opMatrix(A); end
          if isa(B,'numeric'), B = opMatrix(B); end
          
          % Check that the input operators are valid.
          if ~( isa(A,'opSpot') && isa(B,'opSpot') )
             error('One of the operators is not a valid input.')
          end
          
          % Check operator consistency and complexity
          [mA, nA] = size(A);
          [mB, nB] = size(B);
          compatible = isscalar(A) || isscalar(B) || nA == mB;
          if ~compatible
             error('Operators are not compatible in size.');
          end
          op = op@opSpot('FoG', mA, nB);
          op.cflag    = A.cflag  | B.cflag;
          op.linear   = A.linear | B.linear;
          op.children = {A, B};
          op.precedence = 2;
       end
      
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % Display
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function str = char(op)
          % Get operators
          op1 = op.children{1};
          op2 = op.children{2};
          
          % Format first operator
          str1 = char(op1);
          if op1.precedence > op.precedence
             str1 = ['(',str1,')'];
          end
          
          % Format second operator
          str2 = char(op2);
          if op2.precedence > op.precedence
             str2 = ['(',str2,')'];
          end
          
          % Combine
          str = [str1, ' * ', str2];
       end
    end % Methods
       
 
    methods ( Access = protected )
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       % Multiply
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function z = multiply(op,x,mode)
           if mode == 1
              y = apply(op.children{2},x,mode);
              z = apply(op.children{1},y,mode);
           else
              y = apply(op.children{1},x,mode);
              z = apply(op.children{2},y,mode);
           end
        end
    end % methods
   
end
    
