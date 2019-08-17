#include<iostream.h>
#include<conio.h>
void ArrEnter(int P[][3],int ,int );
void ArrEnter(int P[],int );
void ArrShow(int P[],int );
void ArrShow(int P[][3],int ,int );
void Transpose(int P[][3],int ,int );
void Determinant(int P[][3]);
void SumRD(int P[][3],int ,int );
void SumLD(int P[][3],int ,int );
void LSort(int P[],int );
void rev_row(int P[][3],int ,int );


void ArrEnter(int P[],int N)
{       for(int i = 0;i<N;i++)
	{  cout<<"the element"<<i+1<<":";
	    cin>>P[i];
	    cout<<endl;
		}
}
void ArrEnter(int P[][3],int N,int M)
{       for(int i = 0;i<N;i++)
	{	for(int j=0;j<M;j++)
		{  cout<<"the element"<<i+1<<","<<j+1<<":";
		   cin>>P[i][j];
		   cout<<endl;
		}
	}
	clrscr();
}

void ArrShow(int P[][3],int N ,int M)
{
	for(int i = 0;i<N;i++)
	{	for(int j=0;j<M;j++)
		{
		   cout<<P[i][j]<<" ";

		}
		cout<<endl;
	}

}
void ArrShow(int P[],int N )
{
	for(int i = 0;i<N;i++)
	{
		   cout<<P[i]<<" ";
	}
	cout<<endl;

}
void Transpose(int P[][3],int N,int M)
{
	for(int i = 0;i<N;i++)
	{	for(int j=0;j<M;j++)
		{
		   cout<<P[j][i]<<" ";

		}
		cout<<endl;
	}


} /*
void Determinant(int P[][3])
{     int det=0;
	for (int i=0;i<N;i++)
	{	for (int j = 0;j<M;j++)
		 {	for (int k=0;k<M;k++)
			{
			  // det = det + P[0][i]*(P[i+1][]-P[][]);
			}
		}
	}

	cout<<"The determinant of array is :"<<det;
}   */
void SumLD(int P[][3],int N,int M)
{    int sum=0;
	for (int i=0;i<N;i++)
	{      sum += P[i][i];
	}
	cout<<"The Sum of Left Diagonal is :"<<sum<<endl;
}
void LSort(int P[],int N)
{     int temp;
       for(int j=0;j<N;j++)
	for(int i=0;i<N;i++)
	{  if(P[i] > P[i+1])
	   {
		temp = P[i];
		P[i] = P[i+1];
		P[i+1] = temp;
	   }

	}


}
void rev_row(int P[][3],int N , int M)
{
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<M;j++)
			{      if(j<N){
				int temp = P[i][j];
				P[i][j] = P[N-1][j] ;
				P[N-1][j] = temp;
				}
			}

	}

}


void main()
{ 	clrscr();
	int P[3][3];
	 ArrEnter(P,3,3);
	 rev_row(P,3,3);
	 ArrShow(P,3,3);
	getch();
}