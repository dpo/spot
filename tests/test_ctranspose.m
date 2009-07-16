function test_ctranspose
%test_ctranspos  Unit tests for operator transpose

   seed = randn('state');
   
   % Set up matrices and operators for problems
   A  = randn(2,3) + sqrt(-1) * randn(2,3);
   B  = opMatrix(A);
   c  = randn(1,1) + sqrt(-1) * randn(1,1);
   A  = A * c;
   B  = B * c;
   xr = randn(3,2);
   xi = sqrt(-1) * randn(3,2);
   x  = xr + xi;

   % Check operator products
   assertElementsAlmostEqual( ...
      A' * x'  ,...
      B' * x'  );
   assertElementsAlmostEqual( ...
      A' * xr'  ,...
      B' * xr'  );
   assertElementsAlmostEqual( ...
      A' * xi'  ,...
      B' * xi'  );
   assertElementsAlmostEqual( ...
      B'' * x   ,...
      B   * x   );
   assertElementsAlmostEqual( ...
      (x'*B')' ,...
      B* x   );
   
end