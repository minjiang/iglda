function [W, K, n1, n2]=getWSS(Xs, Xt, mu, lambda, dim, kind, p1, p2, p3, labels1)
%SSTCA方法
%原文中的gamma直接在下文中设置
%Xs：源域数据
%Xt：目标域数据，与Xs行数相同
%mu：平衡因子，越大越重视映射后的相似度，越小越重视W的复杂度
%lambda：平衡因子，越大越重视映射前后数据相似性的保持，越小越重视W的复杂度
%dim：当dim为大于等于1的整数时，dim为降维的目标维数；
%     当dim为大于0小于1的小数时，所取特征向量对应的特征值的和>=全部特征值加和*dim
%kind：核函数选择:'Gaussian'、'Laplacian'、'Polynomial',其他一律返回-1
%p1,p2,p3：核函数所要附带的参数
%W：变换矩阵n1+n2->dim
%K：待变换矩阵
%n1,n2：源数据，目标数据的数目

    s1=size(Xs);
    s2=size(Xt);
    n1=s1(2);n2=s2(2);
    
%%%%%%%%%%% 计算K %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X(:,1:n1)=Xs;
    X(:,n1+1:n1+n2)=Xt;
    for i=1:n1+n2 
        for j=i:n1+n2 
            K(i,j)=getKernel(X(:,i), X(:,j), kind, p1, p2, p3);
            K(j,i)=K(i,j);
        end
    end
    
%%%%%%%%%%% 计算L %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L(1:n1, 1:n1)=ones(n1, n1)/(n1*n1);
    L(n1+1:n1+n2, n1+1:n1+n2)=ones(n2, n2)/(n2*n2);
    L(1:n1, n1+1:n1+n2)=ones(n1, n2)/(-n1*n2);
    L(n1+1:n1+n2, 1:n1)=ones(n2, n1)/(-n1*n2);
    
%%%%%%%%%%% 计算H %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H=eye(n1+n2)-ones(n1+n2, n1+n2)/(n1+n2);
    
%%%%%%%%%%% 计算LL，即graph Laplacian matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%k_near：为最近邻的个数
%sigma：当前设置下，最近邻的k_near个直接设为1，即sigma为无穷大
    k_near=floor((n1+n2-1)*0.1);
    sigma=1;
    M=zeros(n1+n2);
    for i=1:n1+n2
        for j=1:n1+n2
            %M(i,j)=K(i,i)+K(j,j)-K(i,j)-K(j,i);
            M(i,j) = sum((X(:,i) - X(:,j)).^2);
        end
        M(i,i)=inf;
        [~,ind]=sort(M(i,:));
        M(i,ind(k_near+1:n1+n2))=0;
        M(i,ind(1:k_near))=1;
%         M(i,ind(1:k_near))=exp(-M(i,ind(1:k_near))/2/sigma);
    end
    M=M';
    D=diag(sum(M,2));
    
    LL=(D-M)/(n1+n2)^2;
    
%%%%%%%%%%% 计算K_yy %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    gamma=0.5;
    K_yy=zeros(n1+n2);
    for i=1:n1
        for j=i:n1
            K_yy(i,j)=getKernel(labels1(:,i), labels1(:,j), 'Polynomial', 1, 0, 1);
            K_yy(j,i)=K_yy(i,j);
        end
    end
    K_yy=gamma*K_yy+(1-gamma)*eye(n1+n2);  

%%%%%%%%%%% 计算W %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Temp=(K*(L+lambda*LL)*K+mu*eye(n1+n2))^(-1)*(K*H*K_yy*H*K);
    [V,D]=eig(Temp);
    D=diag(D);
    D=real(D);
    [D,I]=sort(D,'descend');
    
    if dim>0 && dim<1
        count=1;
        cur=0;
        s=sum(D);
        while cur/s<dim && D(count)>0
            cur=cur+D(count);
            count=count+1;
        end
    else
        count=dim+1;
    end
    
    I=I(1:count-1,1);
    W=V(:,I');
    W=real(W);

    
