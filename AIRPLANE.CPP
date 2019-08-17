#include <stdio.h>
#include <iostream.h>
#include <airplane.h>
#include<string.h>
	Airplane::Airplane(const char* _name, int _type) :
	 type(_type),
	 status(ONRAMP),
	 speed(0),
	 altitude(0),
	 heading(0)
		 {
		 switch (type) {
			 case AIRLINER : ceiling = 35000; break;
			 case COMMUTER : ceiling = 20000; break;
			 case PRIVATE : ceiling = 8000;
			}
	name = new char[50];
	strcpy(name, _name);
	 }
 //
 //
 Airplane::~Airplane()
 {
 delete[] name;
 }
 //
 // Gets a message from the user.
 //
 bool Airplane::SendMessage(int msg, char* response,int spd, int dir, int alt)
 {
 //
 // Check for bad commands.
 //
	 if (spd > 500) {
	strcpy(response, "Speed cannot be more than 500.");
	 return false;
	 }
	 if (dir > 360) {
	 strcpy	(response, "Heading cannot be over 360 degrees.");
	 return false;
	 }
	 if (alt < 100 && alt != -1) {
	 strcpy(response, "I'd crash, bonehead!");
	 return false;
	 }
	 if (alt > ceiling) {
	strcpy(response, "I can't go that high.");
	 return false;
	 }
 //
 // Do something base on which command was sent.
 //
	 switch (msg) {


	 case MSG_TAKEOFF : {
 // Can’t take off if already in the air!
		 if (status != ONRAMP) {
			 strcpy(response, "I'm already in the air!");
			 return false;
		 }
		 TakeOff(dir);
		 break;
	 }
	 case MSG_CHANGE : {
 // Can’t change anything if on the ground.
		 if (status == ONRAMP) {
			 strcpy(response, "I'm on the ground");
			return false;
		 }
 // Only change if a non-negative value was passed.
		 if (spd != -1) speed = spd;
		 if (dir != -1) heading = dir;
		 if (alt != -1) altitude = alt;
		 status == CRUISING;
		 break;
	 }
	 case MSG_LAND : {
		 if (status == ONRAMP) {
			 strcpy(response, "I'm already on the ground.");
			 return false;
		 }
		 Land();
		 break;
	 }
	 case MSG_REPORT : ReportStatus();
	 {
 //
 // Standard reponse if all went well.
 //
		strcpy(response, "Roger.");
		 return true;
	 }
 //
 // Perform takeoff.
 //
      void Airplane::TakeOff(int dir)
	 {
	 heading = dir;
	 status = TAKINGOFF;
	}
 //
 // Perform landing.
 //
	 void Airplane::Land()
	{  }
 return 0;
 }