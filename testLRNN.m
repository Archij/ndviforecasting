function Y = testLRNN(model,X)

X = con2seq(X);
Y = model(X,'useParallel','yes');

ek=seq2con(Y);
Y=ek{1,1};


end

    
       
