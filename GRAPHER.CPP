/************************************
 this is a program to print the graph
	of given equation.
 ************************************
 created by:-Ausaaf Nabi
 created on:-
 ************************************
 all rights reserved
 ************************************/



#include <iostream.h>
#include <conio.h>
#include <graphics.h>
#include <stdlib.h>

void main()
{
 int gdriver = DETECT,gmode,errorcode;
 int maxx,maxy;
 int coffx,coffy,cnst;
 int choice;

 initgraph(&gdriver,&gmode,"");
 errorcode = graphresult();

 if(errorcode != grOk)
 {
	cout<<"there is an error" ;
	exit(0);
	getch();
  }

  setcolor(getmaxcolor());
  maxx = getmaxx();
  maxy = getmaxy();

   if(errorcode == grOk)
 {
   cout<<"\n*************************";
   cout<<"\n Welcome to grapher 3.0";
   cout<<"\n*************************";
   cout<<"\n\nWhich equation do you want to graph?";
   cout<<"\n1: Linear equation";
   cout<<"\n2: quadratic equation";
   cout<<"\n Enter your choice: ";
   cin>>choice;



   switch(choice)
   {
	case 1:{
		 cout<<"\n************************";
		 cout<<"\nEnter coffecient of x: ";
		 cin >>coffx;
		 cout<<"\nEnter coffecient of y: ";
		 cin >>coffy;

		 cout<<"\nEnter constant term: ";
		 cin>>cnst;
		 cout<<"\nEntered equation is:"<<coffx<<"x"<<"+"<<coffy<<"y"<<"+"<<cnst<<"= 0";

		 line(maxx-10,230,330,230);
		 line(475,maxy-50,475,50);
		 getch();
	       break;
	}
	case 2:{
	cout<<"\n***********************";
	cout<<"\nUNDER CONSTRUCTION!\nCOMING TO YOU IN SHORTWHILE.";
	  break;
	}



   }





 }


getch();
}