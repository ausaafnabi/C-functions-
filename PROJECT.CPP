
#include<fstream.h>
#include<conio.h>
#include<stdio.h>
#include<process.h>
#include<dos.h>
#include<stdlib.h>
/********************************************************/
/*   		 Workshop Management System		*/
/********************************************************/
//made by- Ausaaf nabi
//lacks  multiple file handling support

int temp;        //Variable To Increase Lines In Displaying Records.
//DECLARING CLASSES.
class customer
{
   int c_code,dd,mm,yy,order;
   char name[20],city[10],number[10];
public:
   void get_cust();
   void show_cust(int);
   void update_cust();
   void cust_rec();
   int ret_c(){return c_code;}
   void order1(){order++;}
   void new_account();
}c;                         //Declaring object of class

class order                 //For orders.
{
   int c_code,p_code,dd,mm,yy;
   float price;
   long ref;
   char quality;
public:
   order(){ ref=1000;};
   void get_order();
   void show_order(int);
   void profit_sale();
   void profit_month();
}o;

class stock            //For stock details
{
   float price;
   int s_code,dd,mm,yy;
public:
   void get_stock();
   void show_stock(int);
   void update_stock();
}s;

class overhead_exp     //for overhead expenses
{
    int e_code,dd,mm,yy;
    float amount;
public :
    void get_exp();
    void show_exp(int);
    void add_exp();
    float get_amt(){ return amount;}
}e;

class product          //For products.
{
   int p_code;
   char p_name[12],quality;
   float price;
public:
   void get_product();
   void add_product(); //to add details of product to file.
   void show_product(int);
   void show_plist();
   char get_quality(){return quality;}
   void delete_product();
   void update_product();
   int get_p_code() { return p_code;}
   float get_price() { return price;}
}p;

//Class member functions definitions.

void customer::get_cust()
{
 cout<<"\n  Set customer code = ";
 cin>>c_code;
 cout<<"\n  Enter Date (DD/MM/YYYY)= ";
 cin>>dd;
 gotoxy(31,4);
 cout<<"MM = ";   cin>>mm;
 gotoxy(39,4);
 cout<<"  YYYY = "; cin>>yy;
 cout<<"\n  Enter Customer Name = ";
 gets(name);
 cout<<"\n  Enter city =";
 gets(city);
 cout<<"\n  Enter Mobile number = ";
 gets(number);
}

void customer::show_cust(int temp)    //to display customer records.
{
 if(temp==0)
 {
  gotoxy(2,2);    cout<<"CODE";
  gotoxy(11,2);   cout<<"DATE";
  gotoxy(22,2);   cout<<"NAME";
  gotoxy(33,2);   cout<<"CITY";
  gotoxy(42,2);   cout<<"MOBILE NUMBER";
  gotoxy(57,2);   cout<<"ORDERS\n";
  for(int i=0;i<63;i++)
   cout<<"-";
 }
 if(temp>=0)
 {
  gotoxy(2,4+temp);  cout<<c_code;
  gotoxy(8,4+temp);  cout<<dd<<"/"<<mm<<"/"<<yy;
  gotoxy(20,4+temp); cout<<name;
  gotoxy(33,4+temp); cout<<city;
  gotoxy(43,4+temp); cout<<number;
  gotoxy(58,4+temp); cout<<order;
 }
}

void customer::cust_rec()           //to show records of customer.
{
 int code;
 temp=0;
 ifstream fin("customer.dat",ios::binary|ios::in);
 cout<<"\n\n  Enter the Customer code to show it's details =";
 cin>>code;
 while(!fin.eof())
 {
  fin.read((char*)&c,sizeof(c));
  if(c_code==code)
    c.show_cust(temp);
 }

 getch();
 fin.close();
}
void customer::update_cust()       //to update customer.
{
 long code,pos,flag=0;
 temp=0;
 customer c1;
 b:cout<<"\n\n  Enter customer code to update =";
 cin>>code;
 fstream fin("customer.dat",ios::binary|ios::in|ios::out);
 while(!fin.eof())
 {
  fin.read((char*)&c,sizeof(c));
  if(c.c_code==code)
  {
   pos=fin.tellg()-sizeof(c);
   c.show_cust(temp);
   cout<<"\n\n  Press any key to enter new values.";
   getch();
   clrscr();
   c1.get_cust();
   fin.seekp(pos);
   fin.write((char*)&c1,sizeof(c1));
   flag++;
   break;
  }
 }
 if(flag>0)
   cout<<"\n  Information successfully updated..";
 else
 {
   cout<<"\n\tCode Does not Match with any record.";
   goto b;
 }
 fin.close();
 getch();
}
void customer::new_account()      //to create new customer account.
{
 char ch;
 customer c1;
 a:c.get_cust();
 fstream f("customer.dat",ios::binary|ios::in|ios::out);
 while(!f.eof())
 {
  f.read((char*)&c1,sizeof(c1));
  if(c1.c_code==c.c_code)
  {
    cout<<"\n  Customer already exist..";
    clrscr();
    goto a;
  }
 }
 f.write((char*)&c,sizeof(c));
 f.close();
 cout<<"\n   Data Added Sucessfully..";
 getch();
}

void order::get_order()
{
 int pos;
 ifstream f("order.dat",ios::binary);
 /*
 while(!f.eof())
 {
  f.read((char*)&o,sizeof(o));
 }
 ref=ref+1;
 f.close();          */
 cout<<"\n\n  Enter date(DD/MM/YYYY) DD =";
 cin>>dd;
 gotoxy(35,3);
 cout<<"  MM = " ;
 cin>>mm;
 gotoxy(45,3);
 cout<<"  YYYY = ";
 cin>>yy;
 cout<<"\n  Enter Customer Code = ";
 cin>>c_code;
 cout<<"\n  Enter Product Code = ";
 cin>>p_code;
 cout<<"\n  Enter Quality (A/B/C) =";
 cin>>quality;
 ifstream fin("product.dat",ios::binary);
 while(!fin.eof())
 {
  fin.read((char*)&p,sizeof(p));
  if(o.p_code==p.get_p_code())
    if(o.quality==p.get_quality())
      o.price=p.get_price()+(p.get_price()*0.28);
 }
 fin.close();
 cout<<"\n  Total Price (with Tax) = "<<price;
 ofstream fout("order.dat",ios::binary|ios::app);
 fout.write((char*)&o,sizeof(o));
 fout.close();
 cout<<"\n  Reference number = "<<ref;
 ofstream out("customer.dat",ios::binary|ios::app);
 ifstream in("customer.dat",ios::binary);
 while(!in.eof())
 {
  in.read((char*)&c,sizeof(c));
  pos=out.tellp()-sizeof(c);
  if(o.c_code==c.ret_c())
  {
    c.order1();
    out.seekp(pos);
    out.write((char*)&c,sizeof(c));
    break;
  }
 }
 in.close();
 out.close();
 cout<<"\n\n   Order Succesfully Recorded...";
 getch();
}

void order::show_order(int temp)       //To show the order details.
{
 if(temp==0)
 {
  gotoxy(4,2);
  cout<<"DATE";
  gotoxy(14,2);
  cout<<"CUST. CODE";
  gotoxy(25,2);
  cout<<"PROD. CODE";
  gotoxy(37,2);
  cout<<"PRICE";
  gotoxy(44,2);
  cout<<"REF. NO.";
  gotoxy(53,2);
  cout<<"QUALITY";
  cout<<"\n";
  for(int i=0;i<59;i++)
    cout<<"-";
 }
 if(temp>=0)
 {
  gotoxy(2,4+temp);
  cout<<dd<<"/"<<mm<<"/"<<yy;
  gotoxy(17,4+temp);
  cout<<c_code;
  gotoxy(28,4+temp);
  cout<<p_code;
  gotoxy(37,4+temp);
  cout<<price;
  gotoxy(44,4+temp);
  cout<<ref;
  gotoxy(56,4+temp);
  cout<<quality;
 }
}

void order::profit_month()            //for calculating monthly profit
{
 int d1,m1,y1,d2,m2,y2,n=0;
 float p1=0,profit=0,ex=0;
 temp=0;
 cout<<"  Enter date range for which you want to see records";
 cout<<"\n  FROM : DD =";cin>>d1;
 gotoxy(17,2);
 cout<<"MM =";cin>>m1;
 gotoxy(24,2);
 cout<<"YYYY =";cin>>y1;
 cout<<"  TO : DD =";cin>>d2;
 gotoxy(14,3);
 cout<<"MM =";cin>>m2;
 gotoxy(21,3);
 cout<<"YYYY =";cin>>y2;
 cout<<"\n\n  Press any key to Calculate profit..";
 getch();
 clrscr();
 ifstream f("expense.dat",ios::binary);
 ifstream fin("order.dat",ios::binary);
 while(!fin.eof())
 {
  p1=0;
  fin.read((char*)&o,sizeof(o));
  if(yy>=y1 && yy<=y2)
   if(mm>=m1 && mm<=m2)
    if(dd>=d1 && dd<=d2)
      {
       p1=p1+(14*price)/100;
       profit=profit+p1;
       o.show_order(temp);
       gotoxy(61,2);
       cout<<"PROFIT";
       gotoxy(61,4+temp);
       cout<<p1;         //displaying profit on each sale.
       temp++;
       if(n==18)
       {
	getch();
	clrscr();
	temp=0;
	n=0;
       }
       n++;
      }
 }
 while(!f.eof())
 {
  f.read((char*)&e,sizeof(e));
  ex=ex+e.get_amt();            //for calculating net profit
 }
 cout<<"\n Total Overhead  Expense = "<<ex;
 cout<<"\n Gross Profit ="<<profit;
 cout<<"\n\n  Total Net Profit ="<<profit-ex;
 fin.close();
 f.close();
 getch();
}
void order::profit_sale()
{
 temp=0;
 int d,m,y,flag=0;
 long refer;
 float profit=0;
 ifstream fin("order.dat",ios::binary);
 e:cout<<"\n  Enter date of Sale(DD/MM/YY) DD=";
 cin>>d;
 gotoxy(38,2);
 cout<<"MM = ";
 cin>>m;
 gotoxy(46,2);
 cout<<"YYYY = ";
 cin>>y;
 cout<<"  Enter Reference number of Sale = ";
 cin>>refer;
 while(!fin.eof())
 {
  fin.read((char*)&o,sizeof(o));
  if(d==dd && m==mm && y==yy)
    if(refer==ref)
     { profit=10*(price/100);
       cout<<" Press any key to show record.";
       getch();
       clrscr();
       o.show_order(temp);
       gotoxy(61,2);
       cout<<"PROFIT";
       gotoxy(63,4+temp);
       cout<<profit;
       temp++;
       flag++;
       break;
     }
 }
 fin.close();
 if(flag==0)
 {
   cout<<"\n  No such record found.";
   getch();
   clrscr();
   goto e;
 }
 getch();
}

void stock::get_stock()           //for entering stocks
{
 cout<<"\n      Enter Stock Details:";
 cout<<"\n\n  1.  Raw Material (tacks,adhesive)";
 cout<<"\n  2.  Plywood.";
 cout<<"\n  3.  Wood.";
 cout<<"\n  4.  Others.";
 cout<<"\n\n      Enter Stock code = ";
 cin>>s_code;
 cout<<"      Enter Date (DD/MM/YYYY) :  DD=";
 cin>>dd;
 gotoxy(40,10);
 cout<<"MM = ";  cin>>mm;
 gotoxy(48,10);
 cout<<"YYYY = ";  cin>>yy;
 cout<<"\n     Enter Price of stock left = ";
 cin>>price;
}
void stock::show_stock(int temp)
{
 if(temp==0)
 {
  gotoxy(4,2);
  cout<<"DATE";
  gotoxy(17,2);
  cout<<"STOCK CODE";
  gotoxy(30,2);
  cout<<"STOCK LEFT(PRICE)\n";
  for(int i=0;i<45;i++)
    cout<<"-";
 }
 if(temp>=0)
 {
  gotoxy(3,4+temp);
  cout<<dd<<"/"<<mm<<"/"<<yy;
  gotoxy(21,4+temp);
  cout<<s_code;
  gotoxy(35,4+temp);
  cout<<price;
 }
} 
void stock::update_stock()              //for updating stocks.
{
 char ch;
 c:s.get_stock();
 ofstream fout("stock.dat",ios::binary|ios::app);
 fout.write((char*)&s,sizeof(s));
 fout.close();
 cout<<"\n  Stock Updated..";
 cout<<"\n  Do you want to enter more(Y/N) =";
 cin>>ch;
 if(ch=='Y'||ch=='y')
 {
   clrscr();
   goto c;
 }
 getch();
}

void overhead_exp::get_exp()
{
 cout<<"\n  1. Electricity bill.";
 cout<<"\n  2. Transport.";
 cout<<"\n  3. Labour.";
 cout<<"\n  4. Others.";
 cout<<"\n     Enter Expense code = ";
 cin>>e_code;
 cout<<"\n  Enter Date (DD/MM/YYYY): DD = ";
 cin>>dd;
 gotoxy(36,8);
 cout<<"MM = "; cin>>mm;
 gotoxy(44,8);
 cout<<"YYYY = "; cin>>yy;
 cout<<"\n  Enter amount = ";
 cin>>amount;
 ofstream fout("Expense.dat",ios::binary|ios::app);
 fout.write((char*)&e,sizeof(e));
 fout.close();
 cout<<"\n  Data Added Successfully.";
 getch();
}
void overhead_exp::show_exp(int temp)
{
 if(temp==0)
 {
  gotoxy(7,2);      cout<<"DATE";
  gotoxy(19,2);     cout<<"EXPENSE CODE";
  gotoxy(32,2);     cout<<"AMOUNT\n";
  for(int i=0;i<38;i++)
   cout<<"-";
 }
 if(temp>=0)
 {
  gotoxy(4,4+temp);    cout<<dd<<"/"<<mm<<"/"<<yy;
  gotoxy(24,4+temp);   cout<<e_code;
  gotoxy(33,4+temp);   cout<<amount;
 }
}

void product::get_product()           //to enter details of product.
{
 cout<<"\n\n  Enter Product Code = ";
 cin>>p_code;
 cout<<"\n  Enter Product name = ";
 gets(p_name);
 cout<<"\n  Enter Quality of Product(A/B/C)=";
 cin>>quality;
 cout<<"\n  Enter price for Product = ";
 cin>>price;
}
void product::add_product()
{
 product p1;
 a:
 p.get_product();
 fstream f("product.dat",ios::binary|ios::app);
 f.write((char*)&p,sizeof(p));
 f.close();
 cout<<"\n  Product Successfully added.";
 getch();
}
void product::show_product(int temp)
{
 if(temp==0)
 {
  gotoxy(2,2);cout<<"PRODUCT CODE";
  gotoxy(17,2);cout<<"PRODUCT NAME";
  gotoxy(32,2);cout<<"QUALITY";
  gotoxy(42,2);cout<<"PRICE	";
  gotoxy(1,3);
  for(int i=0;i<48;i++)
   cout<<"-";
 }
 if(temp>=0)
 {
  gotoxy(6,4+temp);cout<<p_code;
  gotoxy(18,4+temp);cout<<p_name;
  gotoxy(35,4+temp);cout<<quality;
  gotoxy(42,4+temp);cout<<price;
 }
}
void product::show_plist()
{
 temp=0;
 ifstream fin("product.dat",ios::binary);
 while(!fin.eof())
 {
  fin.read((char*)&p,sizeof(p));
  p.show_product(temp);
  temp++;
 }
 fin.close();
 getch();
}

void product::update_product()      //to update a product
{
 int pcode,pos,flag=0;
 product p1;
 temp=0;
 d:cout<<"\n\n  Enter Product code of product to be updated = " ;
 cin>>pcode;
 cout<<"\n  Original Information : \n";
 fstream fin("product.dat",ios::binary|ios::in|ios::out);
 while(!fin.eof())
 {
  fin.read((char*)&p,sizeof(p));
  if(pcode==p_code)
  {
   pos=fin.tellg()-sizeof(p);
   clrscr();
   p.show_product(temp);
   cout<<"\n  Press any key to enter new values.";
   getch();
   clrscr();
   cout<<"\n\n  Enter Updated Values : \n";
   p1.get_product();
   fin.seekp(pos);
   fin.write((char*)&p1,sizeof(p1));
   flag++;
   break;
  }
 }
 if(flag>0)
   cout<<"\n  Product Successfully Updated.";
 if(flag==0)
 {  cout<<"\n  Product not Fount .... Try again.";
    goto d;
 }
 fin.close();
 getch();
}

void product::delete_product()        //to delete a product
{
 temp=0;
 ifstream fin("product.dat",ios::binary);
 ofstream fout("temp.dat",ios::binary);
 int pcode,flag=1;
 char ch;
 cout<<"\n\n  Enter code of product which is to be deleted=";
 cin>>pcode;
 while(!fin.eof())
 {
  fin.read((char*)&p,sizeof(p));
  if(p.get_p_code()==pcode)
  {
    cout<<"\n  Showing Product details :\n";
    p.show_product(temp+2);
    cout<<"\n\n  Do you want to delete this product (y/n)=";
    cin>>ch;
    if(ch=='n')
      fout.write((char*)&p,sizeof(p));
    flag=0;
    break;
  }
  else
    fout.write((char*)&p,sizeof(p));
   temp++;
 }
 fin.close();
 fout.close();
 remove("product.dat");
 rename("temp.dat","product.dat");
 if(flag==0)
   cout<<"\n\n   Product Successfully deleted..";
 getch();
}


void stock_report()                  //to show stock report
{
 temp=0;
 ifstream fin("stock.dat",ios::binary);
 while(!fin.eof())
 {
  fin.read((char*)&s,sizeof(s));
  s.show_stock(temp);
  temp++;
 }
 fin.close();
 getch();
}

void exp_list()                    //to show expense list
{
 temp=0;
 int n=0;
 ifstream fin("expense.dat",ios::binary);
 while(!fin.eof())
 {
  fin.read((char*)&e,sizeof(e));
  if(n<18)
    e.show_exp(temp);
  if(n==18)
  {
    getch();
    clrscr();
    e.show_exp(temp);
    n=0;    temp=0;
  }
  temp++;
  n++;
 }
 fin.close();
 getch();
}

void sale_report()
{
 clrscr();
 temp=0;
 int q=1;
 ifstream fin("order.dat",ios::binary);
 while(!fin.eof())
 {
  if(q<22)
  {
   fin.read((char*)&o,sizeof(o));
   o.show_order(temp);
   temp++;
  }
  else
  {
   getch();
   q=0;
  }
  q++;
 }
 getch();
}

void mainmenu()             //for displaying main menu
{
 int ch,op;                //and taking options.
 gotoxy(25,10);
 delay(300);
 cout<<"W O R K S H O P  ";
    sound(130);delay(222);
 delay(600);
 cout<<"M A N A G E M E N T  ";
 sound(200);delay(333);
 delay(300);
 cout<<"S Y S T E M";
 sound(150);delay(444);nosound();
 delay(500);
 gotoxy(6,20);
 cout<<"Press any Key to enter main menu...";
 getch();
 beg:
 clrscr();
 gotoxy(35,2);
 cout<<"MAIN MENU";
 gotoxy(8,4);
 for(int i=0;i<60;i++)
  cout<<"=";
 gotoxy(8,5);
 for(i=0;i<18;i++)
 {
   for(int j=0;j<58;j++)
     if(j==0 || j==57)
       cout<<"||";
     else
       cout<<" ";
   gotoxy(8,5+i);
 }
 gotoxy(8,22);
 for(i=0;i<60;i++)
  cout<<"=";
 gotoxy(25,6);
 delay(300);
 cout<<"1->  Show Product List.";
 gotoxy(25,8);
 delay(300);
 cout<<"2->  New Order.";
 gotoxy(25,10);
 delay(300);
 cout<<"3->  Add/Modify/Delete.";
 gotoxy(25,12);
 delay(300);
 cout<<"4->  Generate Reports.";
 gotoxy(25,14);
 delay(300);
 cout<<"5->  Add Overhead Expenses.";
 gotoxy(25,16);
 delay(300);
 cout<<"6->  Exit.";
 gotoxy(25,18);
 delay(300);
 cout<<"Enter Your Choice : ";
 cin>>ch;
 switch(ch)
 {
   case 1: clrscr();
	   p.show_plist();
	   goto beg;

   case 2: clrscr();
	   o.get_order();
	   clrscr();
	   s.update_stock();
	   goto beg;

   case 3: clrscr();
	   gotoxy(28,3);
	   cout<<"1->  Add new Product.";
	   delay(300);
	   gotoxy(28,5);
	   cout<<"2->  New customer account.";
	   delay(300);
	   gotoxy(28,7);
	   cout<<"3->  Delete Product.";
	   delay(300);
	   gotoxy(28,9);
	   cout<<"4->  Update Product info.";
	   delay(300);
	   gotoxy(28,11);
	   cout<<"5->  Update customer info.";
	   delay(300);
	   gotoxy(28,13);
	   cout<<"6->  Go back.";
	   delay(300);
	   gotoxy(28,15);
	   cout<<"     Enter your choice = ";
	   cin>>op;
	   switch(op)
	   {
	    case 1: clrscr();
		    p.add_product();
		    goto beg;
	    case 2: clrscr();
		    c.new_account();
		    goto beg;
	    case 3: clrscr();
		    p.delete_product();
		    goto beg;
	    case 4: clrscr();
		    p.update_product();
		    goto beg;
	    case 5: clrscr();
		    c.update_cust();
		    goto beg;
	    case 6: goto beg;
	    default: gotoxy(20,12);
	    cout<<"INVALID CODE!";
	   // gotoxy(15,21);
	    cout<<" please enter a valid command";
	    sound(100);
	    delay(666);
	    nosound();
	    goto beg;
	   }

   case 4: clrscr();
	   gotoxy(28,2);
	   cout<<"1->  Stock Report.";
	   delay(300);
	   gotoxy(28,4);
	   cout<<"2->  Sale Report.";
	   delay(300);
	   gotoxy(28,6);
	   cout<<"3->  Customer details.";
	   delay(300);
	   gotoxy(28,8);
	   cout<<"4->  Overhead Expense list.";
	   delay(300);
	   gotoxy(28,10);
	   cout<<"5->  Profit on sale.";
	   delay(300);
	   gotoxy(28,12);
	   cout<<"6->  Monthly Profit.";
	   delay(300);
	   gotoxy(28,14);
	   cout<<"7->  GO back.";
	   delay(300);
	   gotoxy(28,16);
	   cout<<"     Enter your choice : ";
	   cin>>op;
	   switch(op)
	   {
	    case 1: clrscr();
		    stock_report();
		    goto beg;
	    case 2: clrscr();
		    sale_report();
		    goto beg;
	    case 3: clrscr();
		    c.cust_rec();
		    goto beg;
	    case 4: clrscr();
		    exp_list();
		    goto beg;
	    case 5: clrscr();
		    o.profit_sale();
		    goto beg;
	    case 6: clrscr();
		    o.profit_month();
		    goto beg;
	    case 7: clrscr();
		    goto beg;
	    default: gotoxy(20,12);
	    cout<<"INVALID CODE!";
	   // gotoxy(15,21);
	    cout<<" please enter a valid command";
	    sound(100);
	    delay(666);
	    nosound();
	    goto beg;
	   }

   case 5: clrscr();
	   e.get_exp();
	   goto beg;
   case 6: clrscr();
	   gotoxy(30,10);
	   cout<<"T H A N K  Y O U ";
	   sound(120);delay(222);sound(100);delay(222);sound(135);delay(222);nosound();
	   delay(666);
	   exit(0);
   default: gotoxy(20,12);
	    cout<<"INVALID CODE!";
	   // gotoxy(15,21);
	    cout<<" please enter a valid command";
	    sound(100);
	    delay(666);
	    nosound();
	    exit(0);
 }
}

void main()
{
 clrscr();
 mainmenu();
 getch();
}


