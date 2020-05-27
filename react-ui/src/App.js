import React from 'react';
import './App.css';
import Amplify from 'aws-amplify';
import awsconfig from './aws-exports';
import Dashboard from './dashboard/dashboard'
import { withAuthenticator } from 'aws-amplify-react';
import { SemanticToastContainer } from 'react-semantic-toasts';
import 'react-semantic-toasts/styles/react-semantic-alert.css';

Amplify.configure(awsconfig);

function App() {
  return (
    <div className="App">
      <SemanticToastContainer />
      <Dashboard thingName="Garage1" />
    </div>
  );
}

export default withAuthenticator(App, true);