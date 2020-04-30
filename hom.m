function hom(x0,A,b)

%forming A0 with same sparsity pattern but different values
[row0,col0,val0] = find(A);
A0 = sparse(row0,col0,val0*1.006);
A0 = full(A0);

ind = 1;
%homotopy steps
for t = 0:0.2:1
    H = (1-t)*F(x0,A0,1.006*b) + t*F(x0,A,b);
    A1 = (1-t)*A0 + t*A;
    f(ind) = norm(H);

    %solve above eqn by direct method
    J = Ja(x0,A1);
    
    %T = [J -H];
    %rref(T)
    
    s = J\(-H);
    %alpha = ls(x0,A1,b,s,0.9);
    x0 = x0 + s;
    
    ind = ind + 1;
end

plot(linspace(0,1,6),f,'-o')
xlabel('t');
ylabel('residual norm');

% Line search based on backtracking
% s is direction, p is contraction factor
function a = ls(x,A,b,s,p)
alpha = 100;
while 1
    if norm(F(x+alpha*s,A,b)) <= norm(F(x,A,b) + (1e-4)*alpha*Ja(x,A)*s)
        break
    else
        alpha = alpha*p;
    end
end
a = alpha;


% Exact Jacobian matrix
% Jacobian has a form:
% |J1|J2|
% |J3|J4|
% Size of Jacobian is 2nx2n
function J = Ja(x,A)
n = (size(x,1)+1)/2;
x(2*n) = 0;
%forming J1
for i = 1:n
    for j = 1:n
        J1(i,j) = x(i)*A(i,j);
    end
end
for i = 1:n
    J1(i,i) = J1(i,i) + dot(A(i,:),x(1:n,:));
end

%forming J2
for i = 1:n
    for j = 1:n
        J2(i,j) = x(i+n)*A(i,j);
    end
end
for i = 1:n
    J2(i,i) = J2(i,i) + dot(A(i,:),x(n+1:2*n,:));
end

%forming J3
for i = 1:n
    for j = 1:n
        J3(i,j) = x(i+n)*A(i,j);
    end
end
for i = 1:n
    J3(i,i) = J3(i,i) - dot(A(i,:),x(n+1:2*n,:));
end

%forming J4
for i = 1:n
    for j = 1:n
        J4(i,j) = -x(i)*A(i,j);
    end
end
for i = 1:n
    J4(i,i) = J4(i,i) + dot(A(i,:),x(1:n,:));
end

J = [J1 J2; J3 J4];
J = J(:,1:2*n-1);


% Power flow equation 
function out = F(x,A,b)
%initial guess size 2n-1
n = size(x,1);
real_n = (n+1)/2;

%forming on diagonal
for i = 1:real_n
    Z(i,i) = x(i);
end
for j = real_n+1:n
    Z(j-real_n,j-real_n) = Z(j-real_n,j-real_n) + x(j)*1i;
end

%plug in nonlinear eqns
F = Z*A*conj(diag(Z))-b;

%squeeze out the output
R = real(F);
I = imag(F);

out = zeros(n+1,1);
ind = size(A,1);
for i = 1:ind
    out(i) = R(i);
end
for i = ind+1:n+1
    out(i) = I(i-ind);
end
