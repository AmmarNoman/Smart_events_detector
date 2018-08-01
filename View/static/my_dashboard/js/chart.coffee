# Copyright (c) 2014, Blind Motion Project
# All rights reserved.

window.dataPreprocessor = null

class ChartManager
    SpeedCoefficient = 3.6

    constructor: (chartId) ->
        if chartId == undefined
            throw new Error('chartId must not be null')

        @chartId = chartId
        @interval = null
        @data = null
        @lastSelectedTime = null

    setData: (data) ->
        if data == null || data == undefined
            throw new Error('Data is null')

        @data = data

        if @interval != null
            @drawSpecificData()

    setInterval: (interval) ->
        if interval == null || interval == undefined
            throw new Error('Data is null')

        @interval = interval

        if @data != null
            @drawSpecificData()

    getDataForInterval: () ->
        if @interval == null
            console.log('Interval not initialized')
            return

        if @data == null
            console.log('Data not initialized')
            return

        dataset = []

        startTime = @interval.start.getTime()
        endTime = @interval.end.getTime()

        headerIsCurrent = true

        for el in @data
            if headerIsCurrent && typeof el[0] != 'number'
                continue

            headerIsCurrent = false

            timeStr = el[1]
            date = moment(timeStr, 'HH:mm:ss.SSS').toDate()
            date = new Date(date.getTime() % MsecInADay)
            dateTime = date.getTime()

            if dateTime >= startTime && dateTime <= endTime
                dataset.push(clone(el))

        return dataset

    drawSpecificData: () ->
        if @interval == null
            console.log('Interval not initialized')
            return

        if @data == null
            console.log('Data not initialized')
            return

        chartData = []

        dataset = @getDataForInterval()

        if dataPreprocessor != null
            dataset = dataPreprocessor(dataset)

        for el in dataset
            timeStr = el[1]
            date = moment(timeStr, 'HH:mm:ss.SSS').toDate()
            date = new Date(date.getTime() % MsecInADay)

            switch el[0]
                when 1
                    chartData.push({
                        date: date,
                        ax:   el[3],
                        ay:   el[4],
                        az:   el[5]
                    })
                when 4
                    chartData.push({
                        date: date,
                        gx:   el[3],
                        gy:   el[4],
                        gz:   el[5]
                    })
                when 'geo'
                    chartData.push({
                        date:  date,
                        speed: el[9] * SpeedCoefficient
                    })

        fields = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'speed']
        @draw(fields, chartData)

    generateGraphs: (fields) ->

        valueAxisMap = {
            'ax' : 'acc',
            'ay' : 'acc',
            'az' : 'acc',
            'gx' : 'gyro',
            'gy' : 'gyro',
            'gz' : 'gyro',
            'speed' : 'speed'
        }

        colorMap = {
            'ax' : '#FF6600',
            'ay' : '#FCD202',
            'az' : '#B0DE09',
            'gx' : '#FF6600',
            'gy' : '#FCD202',
            'gz' : '#B0DE09',
            'speed' : '#6699FF'
        }

        graphs = []
        for field in fields
            graph = {
                "valueAxis": valueAxisMap[field],
                "lineColor": colorMap[field],
                "lineThickness" : 1,
                "bullet": "round",
                "bulletBorderThickness": 1,
                "hideBulletsCount": 30,
                "title": field,
                "valueField": field,
                "fillAlphas": 0
            }
            graphs.push(graph)

        return graphs

    showCursorAt: (date) ->
        if @chart != undefined && !@mouseOver
            @chart.chartCursor.showCursorAt(date)

    showEvents: (events) ->
        if events != null
            @removeAllEventsFromChart()
            for event in events
                 @addEventToChart(event)

    addEventToChart: (event) ->
        guide = new AmCharts.Guide()
        guide.date = new Date(moment(event.start, "HH:mm:ss").toDate().getTime() % MsecInADay)
        guide.toDate = new Date(moment(event.end, "HH:mm:ss").toDate().getTime() % MsecInADay)
        guide.lineColor = "#CC0000"
        guide.lineAlpha = 1
        guide.dashLength = 2
        guide.inside = true
        guide.labelRotation = 90
        guide.label = EventName[event.type] + ' (' + DirectionName[event.direction] + ')'

        @chart.categoryAxis.addGuide(guide)
        @chart.validateNow()

    removeAllEventsFromChart: () ->
        @chart.categoryAxis.guides = []
        @chart.validateNow()


    draw: (fields, chartData) ->
        if chartData.length == 0
            console.log('Empty chart data')

        @chart = AmCharts.makeChart(@chartId, {
            "type": "serial",
            "theme": "dark",
            "pathToImages": "amcharts/images/",
            "dataProvider": chartData,
            "valueAxes": [{
                "id":"acc",
                "axisColor": "#FF6600",
                "axisThickness": 3,
                "gridAlpha": 0,
                "axisAlpha": 1,
                "position": "left",
                "maximum" : 13,
                "minimum" : -13 - 26
            },
            {
                "id":"gyro",
                "axisColor": "#FCD202",
                "axisThickness": 2,
                "gridAlpha": 0,
                "axisAlpha": 1,
                "offset" : 35,
                "position": "left",
                "maximum" : 0.6 + 1.2,
                "minimum" : -0.6
            },
            {
                "id":"speed",
                "axisColor": "#6699FF",
                "axisThickness": 2,
                "gridAlpha": 0,
                "axisAlpha": 1,
                "offset" : 70,
                "position": "left"
            }],
            "graphs": @generateGraphs(fields),
            "chartScrollbar": {},
            "chartCursor": {
                "cursorPosition": "mouse",
                "categoryBalloonDateFormat" : "JJ:NN:SS.fff"
            },
            "categoryField": "date",
            "categoryAxis": {
                "parseDates": true,
                "minPeriod": "fff",
                "axisColor": "#DADADA",
                "minorGridEnabled": true
            },
            "legend": {
                "horizontalGap": 10,
                "marginLeft" : 5,
                "marginRight" : 5,
                "spacing" : 5,
                "valueText" : "",
                "valueWidth" : 20,
                "markerType" : "circle",
                "markerSize" : 12
            }
        })

        changedHandler = (event, handler) =>
            if event.index != undefined
                @lastSelectedTime = chartData[event.index].date.getTime() -
                    @interval.start.getTime()

        @chart.chartCursor.addListener('changed', changedHandler)

        mouseOverHandler = (event, handler) =>
            @mouseOver = true

        mouseOutHandler = (event, handler) =>
            @mouseOver = false

        $('#' + @chartId).mouseover(mouseOverHandler)
        $('#' + @chartId).mouseout(mouseOutHandler)

        @showEvents(allEvents)


window.ChartManager = ChartManager
