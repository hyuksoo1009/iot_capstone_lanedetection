// Imports
const admin = require('firebase-admin');
const serviceAccount = require('./serviceAccount.json');
const locationData = require('./locationData.json');

// JSON To Firestore
const jsonToFirestore = async () => {
  try {
    console.log('Initializing Firebase');
    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount),
    });
    console.log('Firebase Initialized');

    const firestore = admin.firestore();
    await firestore.collection('locationData').doc('data').set(locationData);
    console.log('Upload Success');
  } catch (error) {
    console.log(error);
  }
};

jsonToFirestore();
