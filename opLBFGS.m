classdef opLBFGS < opSpot
%OPLBFGS   Maintain a limited-memory BFGS approximation.
%
%   opLBFGS(n, mem) creates an n-by-n operator that performs
%   matrix-vector multiplication with a limited-memory BFGS
%   approximation with memory m >= 1.
%
%   By default, the operator acts as an inverse L-BFGS approximation,
%   i.e., its inverse is an approximation of the Hessian. It is used
%   as follows:
%
%   B = opLBFGS(n, mem);
%   B = update(B, s, y);
%   d = - B \ g;          % Apply inverse L-BFGS.
%
%   The operator may also be used in forward mode, i.e., as an
%   approximation to of the Hessian. In this case, the attribute
%   update_forward should be set to true, as forward mode incurs
%   additional computational cost. It is used as follows:
%
%   B = opLBFGS(n, mem);
%   B.update_forward = true;
%   B = update(B, s, y);
%   d = - B \ g;          % Apply inverse L-BFGS.
%   Bx = B * x;           % Apply forward L-BFGS.

%   D. Orban, 2014.

%   Copyright 2009, Ewout van den Berg and Michael P. Friedlander
%   See the file COPYING.txt for full copyright information.
%   Use the command 'spot.gpl' to locate this file.

%   http://www.cs.ubc.ca/labs/scl/spot

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties (SetAccess = private)
        mem;
        s;       % Array of s vectors.
        y;       % Array of y vectors.
        ys;      % Array of s'y products.

        alpha;   % Multipliers (for inverse L-BFGS)

        a;       % Negative curvature components of forward L-BFGS
        b;       % Positive curvature components of forward L-BFGS

        insert;  % Current insertion point.
        scaling;
    end

    properties (SetAccess = public)
        update_forward;  % Whether or not to update forward L-BFGS.
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function op = opLBFGS(n, mem)
       %opLBFGS  Constructor.
          if nargin == 1
             mem = 1;
          end
          if nargin > 2
             error('At most one argument can be specified.')
          end

          % Check if input is an integer
          if ~(isnumeric(mem) || mem ~= round(mem))
             error('Memory parameter must be an integer.');
          end

          % Create object
          op = op@opSpot('L-BFGS', n, n);
          op.cflag  = false;
          op.sweepflag  = true;
          op.mem = max(mem, 1);
          op.s = zeros(n, op.mem);
          op.y = zeros(n, op.mem);
          op.ys = zeros(op.mem, 1);
          op.alpha = zeros(op.mem, 1);
          op.a = sparse(n, mem);
          op.b = sparse(n, mem);
          op.update_forward = false;
          op.insert = 1;
          op.scaling = false;
       end % function opLBFGS
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       function op = set.update_forward(op, val)
          if val & ~op.update_forward
            op.a = zeros(size(op.s));
            op.b = zeros(size(op.s));
          end
          op.update_forward = val;
       end

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       % Must use H = update(H, s, y)...
       % How do you get the syntax H.update(s,y) to work???

       function op = update(op, s, y)
       %store  Store the new pair {s,y} into the L-BFGS approximation.
       %       Discard oldest pair if memory has been exceeded.
         ys = dot(s, y);
         if ys <= 1.0e-20;
           warning('L-BFGS: Rejecting (s,y) pair')
         else

           op.s(:, op.insert) = s;
           op.y(:, op.insert) = y;
           op.ys(op.insert) = ys;

           % Update arrays a and b used in forward products.
           if op.update_forward
             op.b(:, op.insert) = y / sqrt(ys);

             for i = 1 : op.mem
               k = mod(op.insert + i - 1, op.mem) + 1;
               if op.ys(k) ~= 0
                 op.a(:, k) = op.s(:, k);                  % Bk0 = I.
                 for j = 1 : i - 1
                   l = mod(op.insert + j - 1, op.mem) + 1;
                   if op.ys(l) ~= 0
                     op.a(:, k) = op.a(:, k) + (op.b(:, l)' * op.s(:, k)) * op.b(:, l);
                     op.a(:, k) = op.a(:, k) - (op.a(:, l)' * op.s(:, k)) * op.a(:, l);
                   end
                 end
                 op.a(:, k) = op.a(:, k) / sqrt(op.s(:, k)' * op.a(:, k));
               end
             end
           end

           % Update next insertion position.
           op.insert = mod(op.insert, op.mem) + 1;
         end
       end
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function x = double(op)
       %double  Convert operator to a double.

          % Can't do op * eye(n), but can do op \ eye(n).
          e = zeros(op.n, 1);
          x = zeros(op.n);
          for i = 1 : op.n
            e(i) = 1;
            x(:, i) = op * e;
            e(i) = 0;
          end
       end
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end % Methods


    methods ( Access = protected )

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function q = multiply(op, x, mode)
       %multiply  Multiply operator with a vector.
       % See, e.g., Nocedal & Wright, 2nd ed., Procedure 7.6, p. 184.

         if ~op.update_forward
           error('L-BFGS: not using forward mode. Set update_forward = true.');
         end

         a = op.a; b = op.b; ys = op.ys;
         q = x;

         % B = B0 + ∑ (bb' - aa').

         for i = 1 : op.mem
           k = mod(op.insert + i - 2, op.mem) + 1;
           if ys(k) ~= 0
             q = q + (b(:, k)' * x) * b(:, k)- (a(:, k)' * x) * a(:, k);
           end
         end
       end % function multiply
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       function r = divide(op, b, mode)
       %divide  Solve a linear system with the operator.
       % See, e.g., Nocedal & Wright, 2nd ed., Algorithm 7.4, p. 178.

         q = b;
         s = op.s; y = op.y; ys = op.ys; alpha = op.alpha;

         for i = 1 : op.mem
           k = mod(op.insert - i - 1, op.mem) + 1;
           if ys(k) ~= 0
             alpha(k) = (s(:, k)' * q) / ys(k);
             q = q - alpha(k) * y(:, k);
           end
         end

         r = q;
         if op.scaling
           last = mod(op.insert - 1, op.mem) + 1;
           if ys(last) ~= 0
             gamma = ys(last) / (y(:, last)' * y(:, last));
             r = gamma * r;
           end
         end

         for i = 1 : op.mem
           k = mod(op.insert + i - 2, op.mem) + 1;
           if ys(k) ~= 0
             beta = (op.y(:, k)' * r) / ys(k);
             r = r + (alpha(k) - beta) * s(:, k);
           end
         end
       end % function divide
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end % methods

end % Classdef
