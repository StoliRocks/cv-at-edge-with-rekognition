import React from 'react';
import Amplify, { Auth, PubSub } from 'aws-amplify';
import IoT from 'aws-sdk/clients/iot';
import { AWSIoTProvider } from '@aws-amplify/pubsub/lib/Providers';
import { Grid, Segment, Statistic, Header, List } from 'semantic-ui-react';
import Moment from 'react-moment';
import moment from 'moment'
// import { toast } from 'react-semantic-toasts';
import ReactDOM from 'react-dom'
import Config from '../config';

var subscriptions = []

// Apply plugin with configuration
Amplify.addPluggable(new AWSIoTProvider({
    aws_pubsub_region: Config.region,
    aws_pubsub_endpoint: Config.wss_endpoint,
}));

class Dashboard extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            telemetry: {},
            name: this.props.name,
            thingName: this.props.thingName,
            message: null,
            iou: 0.0,
            confidence: 0.0,
            vacancy: null,
            actual_ms_since: 0,
            duration_actual_ms: 0,
            actual_ms: 0,
            last_actual_ms: 0,
            start_time: 0,
            logs: []
        };

        this.subscribe = this.subscribe.bind(this);
    }

    render() {

        var RenderRepoListItem = item =>
            <List.Item>
                <List.Icon name={item.icon} size='large' verticalAlign='middle' />
                <List.Content>
                    <List.Header as='a'>{item.state}</List.Header>
                    <List.Description as='a'><strong>START:</strong> <Moment local>{item.start_dttm}</Moment></List.Description>
                    <List.Description as='a'><strong>END:</strong> <Moment local>{item.end_dttm}</Moment></List.Description>
                    <List.Description as='a'><strong>DURATION:</strong> <Moment duration={item.start_dttm} date={item.end_dttm}/></List.Description>
                    <List.Description as='a'><strong>ACTUAL DURATION:</strong> {moment.utc(moment.duration(item.duration_actual_ms).asMilliseconds()).format("HH:mm:ss")}</List.Description>
                </List.Content>
            </List.Item>

        return (
            <div>
                <Segment>
                    <Grid columns={2} relaxed='very'>

                        <Grid.Column>

                            <Header as='h2' textAlign='center'>
                                <Header.Content>
                                    Insights
                                <Header.Subheader>updated with 10 sec snapshots</Header.Subheader>
                                </Header.Content>
                            </Header>

                            <Statistic.Group widths='three' size='mini'>
                                <Statistic>
                                    <Statistic.Value>{this.state.vacancy}</Statistic.Value>
                                    <Statistic.Label>Status</Statistic.Label>
                                </Statistic>

                                <Statistic>
                                    <Statistic.Value>
                                        <Moment unix fromNow>{this.state.start_time}</Moment>
                                    </Statistic.Value>
                                    <Statistic.Label>Since</Statistic.Label>
                                </Statistic>

                                <Statistic>
                                    <Statistic.Value>
                                        {this.state.actual_ms_since}
                                    </Statistic.Value>
                                    <Statistic.Label>Actual Since</Statistic.Label>
                                </Statistic>

                            </Statistic.Group>

                            <Segment>
                                <Grid columns={1} relaxed='very'>
                                    <Grid.Column>
                                        <List divided relaxed style={{ textAlign: "left" }}>
                                            {this.state.logs.map((item) => RenderRepoListItem(item))}
                                        </List>
                                    </Grid.Column>
                                </Grid>
                            </Segment>

                        </Grid.Column>

                        <Grid.Column>

                            <Header as='h2' textAlign='center'>
                                <Header.Content>
                                    {this.state.thingName}
                                    <Header.Subheader>Mechanic Location X</Header.Subheader>
                                </Header.Content>
                            </Header>

                            <Statistic.Group widths='two' size='mini'>

                                <Statistic>
                                    <Statistic.Value>{this.state.confidence}</Statistic.Value>
                                    <Statistic.Label>Confidence</Statistic.Label>
                                </Statistic>

                                <Statistic>
                                    <Statistic.Value>{this.state.iou}</Statistic.Value>
                                    <Statistic.Label>IoU</Statistic.Label>
                                </Statistic>

                            </Statistic.Group>

                            <div>
                                <canvas ref="canvas" width={768} height={432} />
                            </div>

                        </Grid.Column>

                    </Grid>
                </Segment>
            </div>
        );
    }

    subscribe() {

        // var mainSub = PubSub.subscribe(`${this.state.thingName}/frames`).subscribe({
        var mainSub = PubSub.subscribe('garage/001/bay/02').subscribe({
            next: data => {
                this.setState({ message: data.value });
                this.setState({ iou: data.value.DetectLabels.iou.toFixed(2) });
                this.setState({ confidence: data.value.DetectLabels.confidence.toFixed(2) });
                console.log(data);
                const canvas = ReactDOM.findDOMNode(this.refs.canvas);
                const ctx = canvas.getContext('2d');
                const image = new Image();
                image.src = this.state.message.GeneratePresignedUrl;
                const detectedLabels = this.state.message.DetectLabels.labels;
                image.onload = () => {
                    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

                    if (detectedLabels.length > 0) {
                        ctx.strokeStyle = "#FF0000";
                        ctx.lineWidth = 5;

                        const x = detectedLabels[0].Instances[0].BoundingBox.Left * canvas.width;
                        const y = detectedLabels[0].Instances[0].BoundingBox.Top * canvas.height;
                        const w = detectedLabels[0].Instances[0].BoundingBox.Width * canvas.width;
                        const h = detectedLabels[0].Instances[0].BoundingBox.Height * canvas.height;

                        ctx.strokeRect(x, y, w, h);
                    }

                    const bayJSON = {
                        "BoundingBox": {
                            "Width": 0.75,
                            "Height": 0.90,
                            "Left": 0.10,
                            "Top": 0.10
                        }
                    }

                    const box = bayJSON['BoundingBox']
                    const left = canvas.width * box['Left']
                    const top = canvas.height * box['Top']
                    const width = canvas.width * box['Width']
                    const height = canvas.height * box['Height']

                    ctx.strokeStyle = "#33FFE3";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(left, top, width, height);

                };
            },
            error: error => console.error(error),
            close: () => console.log('Done'),
        });
        subscriptions.push(mainSub);

        var thingShadowSub = PubSub.subscribe(`$aws/things/${this.state.thingName}/shadow/get/accepted`).subscribe({
            next: data => {

                console.log('Message received', data)

                if (data.value.state.reported) {

                    if (data.value.state.reported.start_time) {
                        this.setState({ start_time: data.value.state.reported.start_time / 1000 });
                    }

                    if (data.value.state.reported.vacancy) {
                        this.setState({ vacancy: data.value.state.reported.vacancy });
                    }

                    if (data.value.state.reported.actual_ms) {
                        var diff = moment.utc(moment.duration(data.value.state.reported.actual_ms - this.state.last_actual_ms).asMilliseconds()).format("HH:mm:ss");
                        this.setState({ actual_ms: data.value.state.reported.actual_ms });
                        this.setState({ actual_ms_since: diff });
                    }

                    if (data.value.state.reported.last_actual_ms) {
                        this.setState({ last_actual_ms: data.value.state.reported.last_actual_ms });
                    }
                    
                }
            },
            error: error => console.error(error),
            close: () => console.log('Done'),
        });
        subscriptions.push(thingShadowSub);

        var logsSub = PubSub.subscribe(`garage/001/bay/02/logs`).subscribe({
            next: data => {

                var logs = this.state.logs;

                data.value.start_dttm = new Date(data.value.start_ms).toISOString()
                data.value.end_dttm = new Date(data.value.end_ms).toISOString()
                data.value.id = this.state.logs.length;

                if (data.value.state === 'Vacant') {
                    data.value.icon = 'warehouse';
                } else {
                    data.value.icon = 'truck';
                }

                logs.push(data.value);

                this.setState({ logs: logs });
                console.log('Message received', data)

            },
            error: error => console.error(error),
            close: () => console.log('Done'),
        });
        subscriptions.push(logsSub);

        var thingShadowDeltaSub = PubSub.subscribe(`$aws/things/${this.state.thingName}/shadow/update/accepted`).subscribe({
            next: data => {
                if (data.value.state.reported) {

                    if (data.value.state.reported.vacancy) {
                        this.setState({ vacancy: data.value.state.reported.vacancy });
                    }

                    if (data.value.state.reported.start_time) {
                        this.setState({ start_time: data.value.state.reported.start_time / 1000 });
                    }

                    if (data.value.state.reported.actual_ms) {
                        var diff = moment.utc(moment.duration(data.value.state.reported.actual_ms - this.state.last_actual_ms).asMilliseconds()).format("HH:mm:ss");
                        this.setState({ actual_ms: data.value.state.reported.actual_ms });
                        this.setState({ actual_ms_since: diff });
                    }

                    if (data.value.state.reported.last_actual_ms) {
                        this.setState({ last_actual_ms: data.value.state.reported.last_actual_ms });
                    }

                }

                // console.log('Shadow update received', data)
            },
            error: error => console.error(error),
            close: () => console.log('Done'),
        });
        subscriptions.push(thingShadowDeltaSub);

    }

    unsubscribe() {
        subscriptions.forEach(function (sub) {
            if (sub !== null) {
                sub.unsubscribe();
            }
        });
    }

    componentDidMount() {
        Auth.currentCredentials().then((info) => {
            console.log(info);
            const cognitoIdentityId = info.data.IdentityId;
            console.log(cognitoIdentityId);

            const iot = new IoT({
                apiVersion: '2015-05-28',
                credentials: Auth.essentialCredentials(info),
                region: Config.region
            });
            var params = {
                policyName: `${Config.iot_policy_name}`,
                principal: cognitoIdentityId
            };
            iot.attachPrincipalPolicy(params, function (err, data) {
                if (err) console.log(err, err.stack);
            });
        });
        this.subscribe();

        setTimeout(() => {
            PubSub.publish(`$aws/things/${this.state.thingName}/shadow/get`, {});
        }, 2000);
    }

    componentWillUnmount() {
        this.unsubscribe();
    }
}

export default Dashboard;